import os
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

# 禁用 wandb
os.environ["WANDB_DISABLED"] = "true"

# 解析命令行参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="hfl/chinese-bert-wwm-ext", help="选择预训练模型")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="learning_rate")
parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步长")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")  # 
parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="学习率调度策略")  # 
args = parser.parse_args()

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

# 加载数据集
drive_path = "./data"
dataset = load_dataset("csv", data_files={
    "train": os.path.join(drive_path, "train.tsv"),
    "test": os.path.join(drive_path, "test.tsv"),
}, delimiter="\t")

# Tokenization
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text_a"], padding="max_length", truncation=True, max_length=512), batched=True)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

# 训练参数（）
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=True,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,  # 
    lr_scheduler_type=args.lr_scheduler_type,  # 
)

# 评估函数
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 训练
start_time = time.time()
trainer.train()
end_time = time.time()

# 评估
results = trainer.evaluate()
print(f"训练时间: {end_time - start_time:.2f} 秒")
print(f"显存峰值: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
print(f"Accuracy: {results['eval_accuracy']:.3f}")
