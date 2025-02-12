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
![image](https://github.com/user-attachments/assets/ddae453c-2272-42ae-b23e-7d9a7c505ea5)
