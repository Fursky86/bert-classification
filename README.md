# BERT vs RoBERTa 中文文本分类

## 1. 项目简介
本项目对比了 BERT 和 RoBERTa 在中文文本分类任务上的表现，测试不同 batch size 和模型的影响。

## 2. 环境配置
安装必要依赖：
```bash
pip install -r requirements.txt
```

## 3. 运行示例
```bash
python train.py --model hfl/chinese-bert-wwm-ext --batch_size 8 --epochs 3 --learning_rate 2e-5 --gradient_accumulation_steps 16 --warmup_ratio 0.1 --lr_scheduler_type linear
python train.py --model hfl/chinese-roberta-wwm-ext --batch_size 8 --epochs 3 --learning_rate 2e-5 --gradient_accumulation_steps 16 --warmup_ratio 0.1 --lr_scheduler_type linear
```

## 4.BERT vs RoBERTa 训练对比

| 模型     | 准确率 | 训练时间 | 显存最高占用 | batch_size |  epochs  |learing_rate|gradient_accumulation_steps|warmup_ratio|lr_scheduler_type|
|----------|--------|----------|-------------|------------|--------|------------|----------------------------|------------|--------------|
| BERT     | 94.0%  | 188.90 秒 | 3.39 GB     | 8         |3       |2e-5         |                   16      |    0.1     |linear        |
| BERT     | 92.9%  | 150.30 秒 | 4.97 GB     | 16         |3       |2e-5         |                  16      |    0.1     |linear        |
| BERT     | 95.6%  | 316.02 秒 | 3.39 GB     | 8         |5       |2e-5         |                   16      |    0.1     |linear        |
| BERT     | 93.5%  | 191.52 秒 | 3.39 GB     | 8         |3       |1e-5         |                   16      |    0.1     |linear        |
| BERT     | 92.3%  | 186.27 秒 | 3.39 GB     | 8         |3       |2e-5         |                   32      |    0.1     |linear        |
| BERT     | 92.7%  | 189.25 秒 | 3.41 GB     | 8         |3       |2e-5         |                   16      |    0.05    |linear        |
| BERT     | 93.9%  | 189.42 秒 | 3.41 GB     | 8         |3       |2e-5         |                   16      |    0.1     |cosine        |
| BERT     | 95.0%  | 189.24 秒 | 3.41 GB     | 8         |3       |2e-5         |                   16      |    0.1     |cosine_with_restarts  |
| BERT     | 95.5%  | 305.31 秒 | 3.39 GB     | 8         |5       |2e-5         |                   16      |    0.1     |cosine_with_restarts  |
| BERT     | 96.2%  | 611.13 秒 | 3.39 GB     | 8         |10       |2e-5         |                   16      |    0.1     |cosine_with_restarts  |
| RoBERTa  | 94.3%  | 190.73 秒 | 3.40 GB     | 8         |3       |2e-5         |                   16      |    0.1     |linear        |
| RoBERTa  | 94.8%  | 313.79 秒 | 3.40 GB    | 8         |5       |2e-5         |                   16      |    0.1     |cosine_with_restarts  |
| RoBERTa  | 94.3%  | 316.55 秒 | 3.40 GB    | 8         |5       |2e-5         |                   16      |    0.1     |cosine  |
| RoBERTa  | 95.0%  | 307.99 秒 | 3.41 GB    | 8         |5       |2e-5         |                   16      |    0.1     |linear  |
| RoBERTa  | 93.8%  | 241.61 秒 | 4.97 GB    | 16        |5       |2e-5         |                   16      |    0.1     |linear  |
| RoBERTa  | 93.8%  | 310.63 秒 | 3.41 GB    | 8         |5       |1e-5         |                   16      |    0.1     |linear  |
| RoBERTa  | 94.3%  | 310.56  秒 | 3.40 GB    | 8         |5       |2e-5         |                   16      |    0.2     |linear  |
| RoBERTa  | 95.5%  | 611.67 秒 | 3.40 GB    | 8         |10       |2e-5         |                   16      |    0.1     |cosine_with_restarts  |
| RoBERTa  | 95.1%  | 621.43 秒 | 3.40 GB    | 8         |10       |2e-5         |                   16      |    0.1     |cosine  |
| RoBERTa  | 95.4%  | 619.77 秒 | 3.40 GB    | 8         |10       |2e-5         |                   16      |    0.1     |linear  |
| RoBERTa  | 94.3%  | 611.81 秒 | 3.40 GB    | 8         |10       |1e-5         |                   16      |    0.1     |linear  |


