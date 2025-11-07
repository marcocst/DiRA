Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## Requirement

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 2.0.0   | 2.6.0     |
| torchvision  | 0.15.0  | 0.21.0    |
| transformers | 4.45.0  | 4.50.0    |
| datasets     | 2.16.0  | 3.2.0     |
| accelerate   | 0.34.0  | 1.2.1     |
| peft         | 0.14.0  | 0.15.1    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.8.2     |
| flash-attn   | 2.5.6   | 2.7.2     |

### Hardware Requirement

\* *estimated*

| Method                          | Bits |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)              |  16  |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |

## Getting Started

### Installation

> [!IMPORTANT]
> Installation is mandatory.

#### Install from Source

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Extra dependencies available: torch, torch-npu, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, aqlm, vllm, sglang, galore, apollo, badam, adam-mini, qwen, minicpm_v, modelscope, openmind, swanlab, dev

#### Install from Docker Image

```bash
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```

This image is built on Ubuntu 22.04 (x86\_64), CUDA 12.4, Python 3.11, PyTorch 2.6.0, and Flash-attn 2.7.4.

Find the pre-built images: https://hub.docker.com/r/hiyouga/llamafactory/tags

Please refer to [build docker](#build-docker) to build the image yourself.

<details><summary>Setting up a virtual environment with <b>uv</b></summary>

Create an isolated Python environment with [uv](https://github.com/astral-sh/uv):

```bash
uv sync --extra torch --extra metrics --prerelease=allow
```

Run LLaMA-Factory in the isolated environment:

```bash
uv run --prerelease=allow llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
```

</details>

<details><summary>For Windows users</summary>

#### Install PyTorch

You need to manually install the GPU version of PyTorch on the Windows platform. Please refer to the [official website](https://pytorch.org/get-started/locally/) and the following command to install PyTorch with CUDA support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())"
```

If you see `True` then you have successfully installed PyTorch with CUDA support.

Try `dataloader_num_workers: 0` if you encounter `Can't pickle local object` error.

#### Install BitsAndBytes

If you want to enable the quantized LoRA (QLoRA) on the Windows platform, you need to install a pre-built version of `bitsandbytes` library, which supports CUDA 11.1 to 12.2, please select the appropriate [release version](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels) based on your CUDA version.

```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
```

#### Install Flash Attention-2

To enable FlashAttention-2 on the Windows platform, please use the script from [flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel) to compile and install it by yourself.

</details>

<details><summary>For Ascend NPU users</summary>

To install LLaMA Factory on Ascend NPU devices, please upgrade Python to version 3.10 or higher and specify extra dependencies: `pip install -e ".[torch-npu,metrics]"`. Additionally, you need to install the **[Ascend CANN Toolkit and Kernels](https://www.hiascend.com/developer/download/community/result?module=cann)**. Please follow the [installation tutorial](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/softwareinstall/instg/atlasdeploy_03_0031.html) or use the following commands:

```bash
# replace the url according to your CANN version and devices
# install CANN Toolkit
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C20SPC702/Ascend-cann-toolkit_8.0.0.alpha002_linux-"$(uname -i)".run
bash Ascend-cann-toolkit_8.0.0.alpha002_linux-"$(uname -i)".run --install

# install CANN Kernels
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C20SPC702/Ascend-cann-kernels-910b_8.0.0.alpha002_linux-"$(uname -i)".run
bash Ascend-cann-kernels-910b_8.0.0.alpha002_linux-"$(uname -i)".run --install

# set env variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

| Requirement  | Minimum | Recommend      |
| ------------ | ------- | -------------- |
| CANN         | 8.0.RC1 | 8.0.0.alpha002 |
| torch        | 2.1.0   | 2.4.0          |
| torch-npu    | 2.1.0   | 2.4.0.post2    |
| deepspeed    | 0.13.2  | 0.13.2         |
| vllm-ascend  | -       | 0.7.3          |

Remember to use `ASCEND_RT_VISIBLE_DEVICES` instead of `CUDA_VISIBLE_DEVICES` to specify the device to use.

If you cannot infer model on NPU devices, try setting `do_sample: false` in the configurations.

Download the pre-built Docker images: [32GB](http://mirrors.cn-central-221.ovaijisuan.com/detail/130.html) | [64GB](http://mirrors.cn-central-221.ovaijisuan.com/detail/131.html)

#### Install BitsAndBytes

To use QLoRA based on bitsandbytes on Ascend NPU, please follow these 3 steps:

1. Manually compile bitsandbytes: Refer to [the installation documentation](https://huggingface.co/docs/bitsandbytes/installation?backend=Ascend+NPU&platform=Ascend+NPU) for the NPU version of bitsandbytes to complete the compilation and installation. The compilation requires a cmake version of at least 3.22.1 and a g++ version of at least 12.x.

```bash
# Install bitsandbytes from source
# Clone bitsandbytes repo, Ascend NPU backend is currently enabled on multi-backend-refactor branch
git clone -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd bitsandbytes/

# Install dependencies
pip install -r requirements-dev.txt

# Install the dependencies for the compilation tools. Note that the commands for this step may vary depending on the operating system. The following are provided for reference
apt-get install -y build-essential cmake

# Compile & install  
cmake -DCOMPUTE_BACKEND=npu -S .
make
pip install .
```

2. Install transformers from the main branch.

```bash
git clone -b main https://github.com/huggingface/transformers.git
cd transformers
pip install .
```

3. Set `double_quantization: false` in the configuration. You can refer to the [example](examples/train_qlora/llama3_lora_sft_bnb_npu.yaml).

</details>

### Data Preparation

Please refer to [data/README.md](data/README.md) for checking the details about the format of dataset files. You can use datasets on HuggingFace / ModelScope / Modelers hub, load the dataset in local disk, or specify a path to s3/gcs cloud storage.

> [!NOTE]
> Please update `data/dataset_info.json` to use your custom dataset.

You can also use **[Easy Dataset](https://github.com/ConardLi/easy-dataset)** or **[GraphGen](https://github.com/open-sciencelab/GraphGen)** to create synthetic data for fine-tuning.

### Quickstart

Use the following 3 commands to run LoRA **fine-tuning**, **inference** and **merging** of the Llama3-8B-Instruct model, respectively.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

