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
python train.py --model hfl/chinese-bert-wwm-ext --batch_size 8 --epochs 3
python train.py --model hfl/chinese-roberta-wwm-ext --batch_size 8 --epochs 3
```
## 4.不同参数结果展示
## BERT vs RoBERTa 训练对比
| 模型     | 准确率 | 训练时间 | 显存最高占用 | batch_size |
|----------|--------|----------|-------------|------------|
| BERT     | 93.5%  | 251.24 秒 | 4.60 GB     | 8          |
| RoBERTa  | 94.3%  | 196.05 秒 | 6.57 GB     | 8          |
| RoBERTa  | 93.2%  | 158.30 秒 | 8.15 GB     | 16         |
| RoBERTa  | 91.7%  | 138.73 秒 | 11.31 GB    | 32         |


