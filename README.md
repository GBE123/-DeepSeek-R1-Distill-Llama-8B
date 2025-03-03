# -DeepSeek-R1-Distill-Llama-8B
利用qlora进行微调DeepSeek-R1-Distill-Llama-8B
运行环境:i512400f+内存64+显存16G(3080)
torch: 2.5.1+cu118
transformers: 4.49.0

1、从hf下载DeepSeek-R1-Distill-Llama-8B，大约15G
https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B

2、利用满血版deepseek-r1按以下格式生成200条语料，保存为数据集
 {
    "messages": [
      {"role": "system", "content": "你是一名资深证券分析师"},
      {"role": "user", "content": "如何计算动态市盈率？"},
      {"role": "assistant", "content": "动态市盈率=当前股价/（最新季度每股收益×4）。例如某股现价20元，Q1每股收益0.5元，则动态PE=20/(0.5×4)=10倍"}
    ]
  }

3、编写python基于qlora+gradio框架微调代码

4、指定微调参数，运行开始训练
    4.1.数据集加载并向量化
    4.2.加载基础模型DeepSeek-R1-Distill-Llama-8B
    4.3.微调开始
    4.4.微调完成合并模型DeepSeek-R1-Distill-Llama-8B-STOCK，大约15G
    4.5.使用微调后模型进行对话
    
5、分析和对比基础模型对话效果，是否有改进

6、重复2,4,5步

本次微调的代码已经上传github:
python WT_R1_Distill_Llama_8B.py
依赖库慢慢一个一个的装吧

在低参数，低数据量的情况完成一次微调大约5分钟，本次微调遇到的问题较多，集中在以下方面:

1、数据集的预处理，需要完全按照DeepSeek-R1-Distill-Llama-8B进行处理，数据长度，数据尺寸对齐等问题将引发问题

2、显存不够,在开启int4量化情况下训练5轮，每批次加载6条数据，显存占用14G左右

3、训练完成模型合并需要大约25G显存,只能使用内存处理

4、推理速度比ollama管理下慢很多，就算是ollama运行DeepSeek-R1-Distill-Qwen-14B也比较快。

说明ollama做了不少优化。

5、推理结果直接打印控制台正常，交互到gradio页面存在个别字乱码

6、需要使用流式训练和推理
