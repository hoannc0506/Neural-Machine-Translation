from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

class NMTDataset(Dataset):
    def __init__(self, tokenizer, cfg, data_type="train"):
        super().__init__()
        self.cfg = cfg
        self.src_texts, self.tgt_texts = self.read_data(data_type)

        self.src_input_ids = self.texts_to_sequences(tokenizer, self.src_texts)
        self.labels = self.texts_to_sequences(tokenizer, self.tgt_texts)

    def read_data(self, data_type):
        print(f"Loading {data_type} data") 
        data = load_dataset(
            "mt_eng_vietnamese",
            "iwslt2015-en-vi",
            split=data_type
        )
        src_texts = [sample["translation"][self.cfg.src_lang] for sample in tqdm(data)]
        tgt_texts = [sample["translation"][self.cfg.tgt_lang] for sample in tqdm(data)]
        return src_texts, tgt_texts

    def texts_to_sequences(self, tokenizer, texts):
        data_inputs = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.max_len,
            return_tensors='pt'
        )
        return data_inputs.input_ids

    def __getitem__(self, idx):
        return {
            "input_ids": self.src_input_ids[idx],
            "labels": self.labels[idx]
        }

    def __len__(self):
        return np.shape(self.src_input_ids)[0]