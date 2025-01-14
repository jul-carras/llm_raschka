import re
import torch
from torch.utils.data import Dataset

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.string_to_id = vocab
        self.id_to_string = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;!_"()?\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        ids = [self.string_to_id[token] for token in preprocessed]

        return ids
    
    def decode(self, ids):
        text = " ".join([self.id_to_string[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.string_to_id = vocab
        self.id_to_string = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;!_"()?\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        preprocessed = [item if item in self.string_to_id else '<|unk|>' for item in preprocessed]
        
        ids = [self.string_to_id[token] for token in preprocessed]

        return ids
    
    def decode(self, ids):
        text = " ".join([self.id_to_string[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text
    

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]