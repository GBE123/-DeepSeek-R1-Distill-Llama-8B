# -*- coding: utf-8 -*-
"""
@author: gbe 2025.03.02
"""

# 环境要求：Python 3.10+, PyTorch 2.1+
# 必需安装包：
# pip install -Uqqq bitsandbytes==0.41.3
# pip install -Uqqq transformers==4.38.2 peft==0.9.0 accelerate==0.27.2 trl==0.7.11 datasets
# -*- coding: utf-8 -*-
#pip install -U git+https://github.com/huggingface/trl
#pip install tensorboard
import torch
from datasets import load_dataset
from datasets import IterableDataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from trl import SFTTrainer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training  # 新增关键导入
)
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import os
import gradio as gr
import json
import socket
import webbrowser
import psutil
import torch.cuda as cuda  
import platform  
from transformers import TextStreamer
from peft import PeftModel, PeftConfig
import shutil
import os
import time

os.environ["PYTHONUTF8"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ================== 配置区 ==================
#MODEL_NAME = "./DeepSeek-R1-Distill-Llama-8B"
#OUTPUT_DIR = "./deepseek-14b-qlora-finetune"
LOG_DIR = "./logs"


# 量化配置（4位精度）
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

# QLoRA配置（显存优化版）
def get_lora_config():
    return LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",    # 注意力机制查询投影
            "k_proj",    # 注意力机制键投影
            "v_proj",    # 注意力机制值投影
            "o_proj",    # 注意力输出投影
            "gate_proj", # MLP门控投影 (原w1)
            "up_proj"    # MLP上投影 (原w2)
        ],
        modules_to_save=["norm"],  # 关键调整, "embed_tokens"
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

# 训练参数（16GB显存优化）
def get_training_args(TMP_DIR,num_train_epochs, per_device_train_batch_size,  learning_rate):
    return TrainingArguments(
        output_dir=TMP_DIR,
        num_train_epochs=int(num_train_epochs),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        learning_rate=float(learning_rate),
        #learning_rate=2e-5,
        optim="paged_adamw_8bit",
        fp16=True,
        logging_dir=LOG_DIR,
        logging_steps=20,
        save_strategy="steps",
        save_steps=300,
        report_to="tensorboard",
        warmup_ratio=0.1,  # 延长warmup
        weight_decay=0.01,  # 增加正则化
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        warmup_steps=100
    )

class ConversationProcessor(AutoTokenizer):
    def __init__(self, tokenizer: AutoTokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 配置Llama专用模板
        self.tokenizer.chat_template = """
<|user|>{prompt}<|assistant|>
"""

        # 确保tokenizer的pad_token正确设置
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_example(self, example):
        try:
            if not isinstance(example, dict):
                print(f"错误：接收到非字典类型的样本。样本类型: {type(example)}，样本内容: {example}")
                return {"input_ids": [], "attention_mask": []}

            assert "messages" in example, "缺失messages字段"
            messages = example["messages"]
            assert isinstance(messages, list), "messages必须是列表"

            # 仅保留user和assistant角色
            filtered_messages = [
                msg for msg in messages
                if msg.get("role") in ["user", "assistant"]
            ]

            # 格式化对话内容
            text = ""
            for msg in filtered_messages:
                role = msg["role"]
                content = msg["content"].strip()
                text += f"<|{role}|>{content}"

            # 添加生成提示
            text += "<|assistant|>"

            # 分词
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_overflowing_tokens=False,
                padding=False
            )
            # 转换为torch.Tensor
            input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            
            # 生成labels
            input_ids = tokenized["input_ids"]
            labels = input_ids.copy()
            labels[:-1] = input_ids[1:]
            labels[-1] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels  # 必须包含labels
            }
        except Exception as e:
            print(f"数据处理失败: {e}\n样本: {example}")
            return {"input_ids": [], "attention_mask": []}

# ================== 训练回调 ==================
class MemoryOptimizationCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        batch = super().__call__(examples)
        # 实现动态打包逻辑
        input_ids = [seq for ex in examples for seq in self._pack_sequences(ex["input_ids"])]
        attention_mask = [seq for ex in examples for seq in self._pack_sequences(ex["attention_mask"])]
        # 转换为torch.Tensor
        batch["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        batch["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        return batch

    def _pack_sequences(self, input_ids):
        max_len = self.tokenizer.model_max_length
        return [input_ids[i:i+max_len] for i in range(0, len(input_ids), max_len)]

# 加载数据集（流式）
def get_streaming_dataset(path):
    raw_data = load_dataset("json", data_files=path, streaming=True, encoding="utf-8")["train"]
    return raw_data

# 数据质量检查
def validate_sample(sample):
    try:
        assert "messages" in sample, "缺失对话字段"
        convs = sample.get("messages")
        assert len(convs) >= 1, "对话轮次不足"

        content = " ".join([msg["content"] for msg in sample["messages"]])
        #if len(content.split()) < 15:  # 过滤过短样本
        #    return False
        #if "请详细推导" in content and "用户补充问题" in content: # 过滤含循环模式的样本
        #    return False

        return True
    except AssertionError as e:
        print(f"无效样本: {sample}\n错误: {str(e)}")
        return False

# 流式处理数据集
def process_dataset_streaming(dataset_generator, processor):
    for example in dataset_generator:
        if validate_sample(example):
            processed_example = processor.process_example(example)
            yield processed_example


# ================== 推理测试 ==================
# 构造对话消息
test_messages = [
    {"role": "user", "content": "你是一名资深证券分析师，如何计算动态市盈率？"}
]


generation_config = {
    "max_new_tokens": 2048,  # 增加生成长度
    "temperature": 0.9,      # 提高创造性
    "top_k": 50,             # 增加采样多样性
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 4,  # 禁止4-gram重复
    "do_sample": True,
    "num_beams": 2,          # 使用beam search
    "early_stopping": True,
    "eos_token_id": None,  # 显式指定结束符
    "pad_token_id": None
}
from transformers import DataCollatorForSeq2Seq
def get_data_collator(tokenizer):
    return DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        padding=True,
        return_tensors="pt"
    )

def safe_release(obj):
    """安全释放对象内存"""
    if isinstance(obj, torch.nn.Module):
        # PyTorch模型专用清理
        obj.to('cpu')  # 先转移到CPU
        obj.zero_grad(set_to_none=True)
        for name, child in obj.named_children():
            safe_release(child)
        del obj
    elif hasattr(obj, 'cpu'):
        # 张量转移到CPU后删除
        obj.cpu()
        del obj
    else:
        del obj
    gc.collect()
    torch.cuda.empty_cache()
import ctypes
def full_memory_purge():
    """全栈式内存清理"""
    # PyTorch清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Python垃圾回收
    gc.collect()
    # 系统级内存整理（Linux有效）
    if os.name == 'posix':
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    # Windows系统清理
    elif os.name == 'nt':
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(1024**3), ctypes.c_size_t(1024**3))
def tensor_obliteration():
    """核弹级张量内存清除"""
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    obj.data = obj.cpu().data
                del obj
        except:
            pass
    full_memory_purge()

        
def fn_train(data_file_path,MODEL_NAME,Y_model_path,tmp_model_path,new_model_path,epochs,batch_size,rate):
    #加载数据文件
    #参数指定
    #开始训练
    #保存结果
    #合并模型 
    try:
        print("开始")
        outstr =""
        outstr = f"加载数据集,分词处理....\n"
        yield gr.update(value=outstr)
    
        
        DATASET_PATH = data_file_path.name
        print(DATASET_PATH)
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
    
        processor = ConversationProcessor(tokenizer)
    
        dataset_generator = get_streaming_dataset(DATASET_PATH)
        
        processed_dataset = process_dataset_streaming(dataset_generator, processor)
        # 将生成器的数据收集到列表中
        data_list = []
        for data in processed_dataset:
            data_list.append(data)
        
        # 使用 from_list 创建 Dataset 对象
        train_dataset = Dataset.from_list(data_list)
    

        #train_dataset = IterableDataset.from_generator(
        #    lambda: process_dataset_streaming(dataset_generator, processor)
        #)

        bnb_config = get_bnb_config()
        lora_config = get_lora_config()
        training_args = get_training_args(tmp_model_path,epochs, batch_size, rate)
       
        outstr =  f"加载基础模型....\n"
        yield gr.update(value=outstr)
        
        #print(MODEL_NAME)
        #print(training_args)
        #print(bnb_config)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        outstr =  f"准备模型进行量化4训练....\n"
        yield gr.update(value=outstr)
        # 准备模型进行4位训练
        model = prepare_model_for_kbit_training(model)
    
        # 应用LoRA配置
        model = get_peft_model(model, lora_config)
    
        # 实例化数据收集器
        data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
        
        
        outstr =  f"实例化SFTTrainer....\n"
        yield gr.update(value=outstr)
        # 实例化SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=get_data_collator(tokenizer),#data_collator,
            callbacks=[MemoryOptimizationCallback()]
        )
    
              
        outstr =  f"开始训练....\n"
        yield gr.update(value=outstr)
        # 开始训练
        print(f"开始训练....\n")
        trainer.train()
        print(f"训练完成....\n")
        outstr =  f"训练完成!\n"
        yield gr.update(value=outstr)
        
        # 保存模型增量     
        trainer.save_model(new_model_path)
        tokenizer.save_pretrained(new_model_path)
    
        outstr =  f"训练完成，已保存模型增量...\n"
        print(f"训练完成，模型已保存\n")
        
        yield gr.update(value=outstr)
        
        #释放资源以便进行合并
        del trainer
        del model
        del train_dataset        
        #del data_list
        torch.cuda.empty_cache()
        gc.collect()  

        time.sleep(3)
        yield gr.update(value=outstr)
        
        tmp_mod_path = "./"+tmp_model_path+"/checkpoint-5"
        
        #合并模型        
        peft_config = PeftConfig.from_pretrained(tmp_mod_path, trust_remote_code=True)
        
        outstr = outstr.join(f"加载基础模型\n")
        yield gr.update(value=outstr)
        
        print(f"加载基础模型\n")    
        # 加载基础模型时指定半精度
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            return_dict=True,
            device_map="cpu",
            torch_dtype=torch.float16  # 关键修改：保持半精度
        )
        outstr = outstr.join(f"加载增量模型模型\n")
        yield gr.update(value=outstr)
        print(f"加载增量模型模型\n")
        # 加载并合并LoRA适配器
        
        model = PeftModel.from_pretrained(base_model, tmp_mod_path)
        merged_model = model.merge_and_unload()

        outstr = outstr.join(f"创建目录\n")
        yield gr.update(value=outstr)
        shutil.rmtree(new_model_path)
        os.mkdir(new_model_path)
        outstr = outstr.join(f"保存目录\n")
        yield gr.update(value=outstr)
        print(f"保存时保持半精度并清理存储格式\n")
        # 保存时保持半精度并清理存储格式
        merged_model.save_pretrained(
            new_model_path,
            safe_serialization=True,  # 使用safetensors格式
            max_shard_size="2GB"      # 分片存储
        )
        print(f"同时保存tokenizer\n")
        # 同时保存tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(new_model_path)
        outstr = outstr.join(f"保存最新微调模型完毕\n")
        yield gr.update(value=outstr)
        
    
        del base_model, model, merged_model
        #tensor_obliteration()
        #full_memory_purge()
        # 多级GC
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        outstr = "全部微调工作完成...100%\n"
        print(f"全部微调工作完成...100%\n")
        return outstr
    except Exception as e:
        # 处理异常情况
        error_str = f"训练过程中出现错误: {str(e)}\n"
        outstr =  f"训练过程中出现错误: {e}\n"
        print(error_str)
        yield gr.update(value=outstr)
        return error_str
    finally:
        # 释放资源
        # 删除模型和张量对象
        # 这里由于之前没有实际的模型对象创建，只是模拟，所以注释掉相关代码
        # del model

        if 'trainer' in locals():
            del trainer
        if 'model' in locals():
            del model
        if 'train_dataset' in locals():
            del train_dataset
        if 'data_list' in locals():
            del data_list
        # 释放 GPU 缓存        
        torch.cuda.empty_cache()

        # 删除数据集对象
        # 由于之前没有实际的数据集对象，这里只是示例，实际使用时按需删除
        # del train_dataset

        # 触发 Python 垃圾回收
        gc.collect()
    return f"全部微调工作完成...100%\n"

g_load_model_flag=False
g_model_radio=""
g_ask_model=None
g_ask_tokenizer=None
def fn_ask(radio,MODEL_NAME,Y_model_path,new_model_path,question_input):
    #加载模型
    #模型生成
    #输出结果
    try:
        if g_load_model_flag==False:
            # 模型名称或本地路径
            model_name=Y_model_path
            print(radio)
            if radio=="微调后模型":
                model_name = new_model_path  # 替换为实际的模型路径
            
            print(model_name)
            # 加载分词器
            g_ask_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            load_in_4bit = True
            bnb_4bit_compute_dtype = torch.float16
            bnb_4bit_use_double_quant = True
            bnb_4bit_quant_type = "nf4"
            
                   
            # 使用 int4 量化加载模型
            
            g_ask_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=bnb_4bit_compute_dtype,
                quantization_config={
                "load_in_4bit": load_in_4bit,
                "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
                "bnb_4bit_quant_type": bnb_4bit_quant_type
                }
            )
            # 打印模型结构
            #print(model)
           
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 打印当前显存占用情况
            if torch.cuda.is_available():
                print(f"当前显存占用: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")    
             
        # 输入文本        
        inputs = g_ask_tokenizer(question_input, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)


         
        streamer = TextStreamer(g_ask_tokenizer, skip_prompt=True)  # 跳过重复显示问题
        # 生成文本时使用流式处理
        
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            do_sample=True,         # 启用抽样策略
            top_p=0.9,              # 推荐配合使用核抽样
            temperature=0.7,        # 控制生成随机性
            streamer=streamer,
            num_beams=1,            # 必须设置为1
            eos_token_id=g_ask_tokenizer.eos_token_id
        )
        
        # 在生成时实时流式输出
        output = g_ask_model.generate(**generation_kwargs)
        
        generated_text = ""
        for token_id in output[0]:
            new_token = g_ask_tokenizer.decode(token_id.item(), skip_special_tokens=True)            
            generated_text += new_token
            yield generated_text  # 使用yield逐步返回结果
            time.sleep(0.1)  # 模拟生成延迟
        
        del output,attention_mask,input_ids,inputs

        # 释放 GPU 缓存        
        #torch.cuda.empty_cache()

        #gc.collect()   

    except Exception as e:
        yield(f"加载模型时出错: {e}")
        torch.cuda.empty_cache()
        
    finally:
        # 释放资源
        # 删除模型和张量对象
        # 这里由于之前没有实际的模型对象创建，只是模拟，所以注释掉相关代码
        # del model

        if 'model' in locals():
            del model

        # 释放 GPU 缓存        
        torch.cuda.empty_cache()

        # 删除数据集对象
        # 由于之前没有实际的数据集对象，这里只是示例，实际使用时按需删除
        # del train_dataset

        # 触发 Python 垃圾回收
        gc.collect()
   
def get_system_info():
    
    cpu_model = torch.cuda.get_device_name(0)

    # 获取内存总量（单位：GB）
    memory = psutil.virtual_memory()
    memory_total = memory.total / (1024 ** 3)

    # 获取显存总量（单位：GB）
    if cuda.is_available():
        cuda_total = cuda.get_device_properties(0).total_memory / (1024 ** 3)
    else:
        cuda_total = 0

    return cpu_model, cuda_total, memory_total

# 获取系统信息
cpu_model, cuda_total, memory_total = get_system_info()


#================================程序界面构建=============================================================================
# Gradio界面
html_head = """
<head>
    <meta charset="UTF-8">
    <!-- 其他头部信息 -->
</head>
"""

with gr.Blocks(
        title="hi,GBE今天怎么样",       
        css="""
        .gradio-container {max-width: 2000px !important}
        .answer-box {min-height: 500px !important;}
        .left-panel {padding-right: 20px; border-right: 1px solid #eee;}
        .right-panel {height: 100vh;}
        .wide-row { width: 80%; }
        .green-button {
            background-color: green;
            color: white; 
        }
       .gradio-label {
            font-size: 8px !important;
            font-weight: normal !important;
        }
        .gradio-container input {
            font-size: 8px !important;
        }
        .gradio-container textbox {
            font-size: 8px !important;
        }
        .gray-background textarea, .gray-background input[type="text"] {
        background-color: #cccccc !important;
        }        
        """
        ) as demo:
                     
    gr.HTML(html_head, visible=False)
    gr.Markdown("开始吧...")
    with gr.Row(elem_classes="wide-row"):        
        with gr.Column(scale=10, elem_classes="left-panel"):
            gr.Markdown("## 📝 系统信息及参数配置")
            with gr.Group():                
                data_file_input = gr.File(label="上传json文档", file_types=[".json"])
                with gr.Row():
                    cpu_textbox = gr.Textbox(label="显卡",interactive=False,value=f"{cpu_model}",elem_classes="gray-background")#只读                    
                    xc_textbox = gr.Textbox(label="显存",interactive=False, value=f"{cuda_total:.2f} GB",elem_classes="gray-background")#只读
                    nc_textbox = gr.Textbox(label="内存",interactive=False, value=f"{memory_total:.2f} GB",elem_classes="gray-background")#只读
                    data_file_size_textbox = gr.Textbox(label="数据集文件大小",interactive=False,elem_classes="gray-background")
                with gr.Row():    
                    MODEL_NAME_textbox = gr.Textbox(label="微调模型名称", value="DeepSeek-R1-Distill-Llama-8B")#只读                                     
                    Y_model_path_textbox = gr.Textbox(label="原模型文件路径(文件夹)", value="DeepSeek-R1-Distill-Llama-8B")                
                with gr.Row():    
                    tmp_path_textbox = gr.Textbox(label="临时文件路径(文件夹)",value="DeepSeek-R1-Distill-Llama-8B-tmp")
                    new_model_path_textbox = gr.Textbox(label="新模型文件路径(文件夹)",value="DeepSeek-R1-Distill-Llama-8B-STOCK")
                with gr.Row():    
                    epochs_textbox = gr.Textbox(label="轮次",value="5")#只读
                    batch_size_textbox = gr.Textbox(label="批次大小(每次多少条数据)",value="8")#只读
                    rate_textbox = gr.Textbox(label="学习率",value="2e-5")#只读                
                
                train_btn = gr.Button("开始微调训练", variant="primary")
            gr.Markdown("## ❓ 训练&问答")
            with gr.Group():
                options = ["原模型", "微调后模型"]
                radio = gr.Radio(choices=options, label="请选择模型",value="微调后模型")
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=4,
                    placeholder="例如：本文档的主要观点是什么？",
                    elem_id="question-input"
                )
                ask_btn = gr.Button("🔍 开始提问", variant="primary",  elem_classes="green-button")
                status_display = gr.HTML("", elem_id="status-display")
        with gr.Column(scale=10, elem_classes="right-panel"):
            gr.Markdown("## 📝 训练情况")
            train_output = gr.Textbox(
                label="训练情况",
                lines=4,
                placeholder="例如：本文档的主要观点是什么？",
                elem_id="train-output"
            )
            gr.Markdown("## 📝 答案")
            answer_output = gr.Textbox(
                label="回答",
                interactive=False,
                lines=25,
                elem_classes="answer-box",
                autoscroll=True,
                show_copy_button=True
            )
            gr.Markdown("""
            <div class="footer-note">
                *回答生成可能需要1-2分钟，请耐心等待<br>
                *支持多轮对话，可基于前文继续提问
            </div>
            """)
    train_btn.click(
        fn=fn_train,
        inputs=[data_file_input,MODEL_NAME_textbox,Y_model_path_textbox,tmp_path_textbox,new_model_path_textbox,epochs_textbox,
                batch_size_textbox,rate_textbox],
        outputs=[train_output],
        show_progress="hidden"
    )
    ask_btn.click(
        fn=fn_ask,
        inputs=[radio,MODEL_NAME_textbox,Y_model_path_textbox,new_model_path_textbox,question_input],
        outputs=[answer_output],
        show_progress="hidden"
    )


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0  # 更可靠的检测方式
if __name__ == "__main__":    
    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)
    
    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)
        
    try:    
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port,
            server_name="0.0.0.0",
            show_error=True,
            ssl_verify=False
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")
# ================== 释放资源 ==================
# 删除模型和张量对象
#del model, inputs, outputs
del g_ask_model
# 释放 GPU 缓存
torch.cuda.empty_cache()
# 删除数据集对象
# 触发 Python 垃圾回收
import gc
gc.collect()
