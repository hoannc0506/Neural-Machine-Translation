import os
import numpy as np
import torch

import evaluate
from transformers import (
    MBart50TokenizerFast,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from config import NMTConfig
from dataset import NMTDataset
from utils import validate

# load config
cfg = NMTConfig()

# init model and tokenizer
print("Load model")
tokenizer = MBart50TokenizerFast.from_pretrained(cfg.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model
)

# define metric and compute metric function
metric = evaluate.load("sacrebleu")
compute_metrics = lambda pred: validate(tokenizer, metric, pred)


if __name__ == "__main__":
    # Name wandb project
    os.environ["WANDB_PROJECT"]="En2Vi-Machine-Translation"
    
    # load dataset
    train_dataset = NMTDataset(tokenizer, cfg, data_type="train")
    valid_dataset = NMTDataset(tokenizer, cfg, data_type="validation")
    test_dataset = NMTDataset(tokenizer, cfg, data_type="test")
    
    # define training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy='steps',
        save_steps=cfg.eval_steps,
        eval_steps=cfg.eval_steps,
        output_dir=cfg.ckpt_dir,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        save_total_limit=cfg.save_total_limit,
        num_train_epochs=cfg.num_train_epochs,
        load_best_model_at_end=True,
        report_to="wandb", # wandb logging
        run_name="nmt-en2vi-mBART50" # wandb run name
    )


    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    # finish wandb
    wandb.finish()