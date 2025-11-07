from modelscope import snapshot_download  
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct')
model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3-8B-Instruct',
    cache_dir='/root/autodl-tmp/LLaMA-Factory-main/llama3—model'  # 替换为你的目标路径
)