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
| RoBERTa  | 94.3%  | 188.62 秒 | 3.40 GB     | 8         |3       |2e-5         |                   16      |    0.1     |linear        |
| RoBERTa  | 93.2%  | 158.30 秒 | 8.15 GB     | 16        |
| RoBERTa  | 91.7%  | 138.73 秒 | 11.31 GB    | 32        |
