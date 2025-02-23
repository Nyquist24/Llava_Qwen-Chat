# Llava (CLIP+Qwen1.5/2): Towards GPT Level Vision and Speech Interaction [.Git](https://github.com/Nyquist24/Llava_Qwen-Chat)

## Progress
* **`2024.12.05`** 🌟Code of Pretrain: train_qwen, builder, llava_qwen (LlavaQwen2ForCausalLM, ~~AutoTokenizer.register()~~)
* **`2024.12.23`** 🌟Pretrain in [liuhaotian/LLaVA-Pretrain](https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain)
* **`2024.01.04`** 🌟Add [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) of OpenCompass
* **`2025.01.06`** 🌟SFT(Full parameter and Lora) in [llava_v1_5_mix665k](https://hf-mirror.com/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)
* **`2025.02.03`** 🌟Different build_prompt for Different mission(VQA, Yes or No, OCR, MME)
* **`2025.to do`** 🌟Inference in pure C/C++ (llama.cpp)



## Contents
- Llava (CLIP+Qwen1.5/2): Towards GPT Level Vision and Speech Interaction
  - [🔥 Progress](#progress)
  - [⭐ Training](#training)
    - [Requirements and Installation](#requirements-and-installation)
    - [Data Preparation](#data-preparation)
    - [Finetune](#finetune)
  - [📏Evaluating on Benchmarks](#evaluating-on-benchmarks)
    - [VLMEvalKit](#vlmevalkit)
  - [👍 Acknowledgement](#-acknowledgement)



##  Training
### Requirements and Installation
```
git clone https://github.com/Nyquist24/Llava_Qwen-Chat.git
conda create -n llavaqwen python=3.10 -y
conda activate llavaqwen
pip install --upgrade pip
pip install -e .
pip install flash-attn --no-build-isolation
```

### Data Preparation
**Pretrain:** [liuhaotian/LLaVA-Pretrain](https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain)
**Finetune:** [liuhaotian/LLaVA-Instruct-150K](https://hf-mirror.com/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main) refer to [Llava](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning)
- Put the pretrain data in your data/, Organize the finetune data as follows in:
```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

### Pratrain

According to llava-series paper, in Pretrain phase, **CLIP** and **Qwen** should be Freezed

```
sh pretrain_qwen2.sh
```
**Tips:** 
- the **--output_dir** in pretrain_qwen2.sh must contrains 'llava' and 'qwen' (for VLMEvalKit)
- the **--tune_mm_mlp_adapter** should be set True
- checkpoints will be saved in ./checkpoints

### Finetune

```
sh finetune_qwen.sh
```



## 📏Evaluating on Benchmarks
### VLMEvalKit
Add the model path of `llava_qwen2` in `VLMEvalKit/vlmeval/config.py`, like:
```
llava_series = { 
    'llavaqwen': partial(LLaVAQwen, model_path='/root/ ... /checkpoints/llava_qwen1.5-4B-Chat'),
    ...
}
```

Follow the [instuctions in VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) to set the GPT as the judge model.


Then configure the `.env` file in the `VLMEvalKit` folder:
```
OPENAI_API_KEY=sk-
OPENAI_API_BASE=
```
Evaluating on Different benchmarks:
```
CUDA_VISIBLE_DEVICES=0 python run.py --data AI2D_TEST OCRBench ... --model llavaqwen --verbose
```

## 👍 Acknowledgement
Thanks to the following outstanding works: [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [Qwen-2.5](https://github.com/QwenLM/Qwen2.5),[VITA](https://github.com/VITA-MLLM/VITA)  and [VLMEvalkit](https://github.com/open-compass/VLMEvalKit).
