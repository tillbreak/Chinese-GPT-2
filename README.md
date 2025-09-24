# Chinese-GPT-2

## 1. 简介

采用 40MB 的科幻小说集作为数据集，从头训练一个 82M 的中文 GPT-2 模型。初始模型经 12 个周期的训练后，最终可以续写 10 句以上的科幻小说。

## 2. 快速开始

### 2.1 环境

首先，下载模型训练所需依赖：

```bash
pip install -r requirements.txt
```

### 2.2 数据集

准备中文语料，并放置在 `./data/` 文件夹下。注意，放置的输入语料需事先转换为一定格式的 `.json` 文件（参考格式见 `example.json`）。

### 2.3 模型配置

在 `model_config` 定义初始 GPT-2 模型的超参数配置：

- `initializer_range`: 0.02  
  定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为 0，标准差为 0.02 的正态分布中进行随机初始化。
- `layer_norm_epsilon`: 1e-05  
  用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为 1e-05，用于稳定训练。
- `n_ctx`: 512  
  表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为 1024，即模型一次最多能处理 1024 个 token。
- `n_embd`: 768  
  表示每个 token 的嵌入维度大小，即模型中词向量的维度。设置为 768，即每个词汇的表示向量是 768 维的。
- `n_head`: 8  
  表示自注意力机制中的注意力头的数量。设置为 12，即模型的多头注意力机制中有 12 个独立的头。
- `n_layer`: 10  
  表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
- `n_positions`: 512  
  表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 `n_ctx` 一致，表示模型最多能处理 1024 个位置的 token。
- `vocab_size`: 13317  
  表示词汇表的大小，即模型可以识别和生成的词汇数量。

### 2.4 训练

使用处理好的数据集来训练初始 GPT-2 模型：

```bash
python train.py \
  --device 0,1 \
  --model_config config/model_config_small.json \
  --tokenizer_path cache/vocab_small.txt \
  --raw_data_path data/train.json \
  --tokenized_data_path data/tokenized_cov512/ \
  --raw \
  --epochs 12 \
  --batch_size 12 \
  --gradient_accumulation 2 \
  --stride 512 \
  --num_pieces 32 \
  --min_length 128 \
  --lr 8e-5 \
  --warmup_steps 3000 \
  --log_step 48 \
  --output_dir model_full_cov512/ \
  --writer_dir runs/full_cov512/
```

训练过程中，每个 epoch 对应的模型将存储在 `./output_dir` 目录下，最终训练好的模型将存储在 `./output_dir/final_model/` 路径中。

### 2.5 生成

使用 `generate.py` 脚本对训练后的模型进行测试：

```bash
python generate.py \
  --device 0 \
  --length 350 \
  --tokenizer_path cache/vocab_small.txt \
  --model_path model/final_model \
  --prefix "[CLS]哈利与赫敏" \
  --save_samples \
  --save_samples_path ./mnt/
```

### 2.6 备注

本项目的目录结构如下：

- `cache`: 存放词表
- `config`: 存放初始模型超参数
- `data`: 存放训练语料
- `logs`: 存放训练日志
- `mnt`: 存放模型生成样本
- `model_full_cov512`: 存放训练后的模型
- `runs`: 存放模型训练时的 loss 变化记录
- `tokenizations`: 存放分词器