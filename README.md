# DiRA

## Getting Started


```bash
git clone --depth 1 https://github.com/marcocst/DiRA.git
cd DiRA
pip install -e ".[torch,metrics]" --no-build-isolation
```

Extra dependencies available: torch, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, aqlm, vllm, sglang, galore, apollo, badam, adam-mini, qwen, minicpm_v, modelscope, openmind, swanlab, dev


Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```
