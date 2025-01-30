import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


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
    

class SelfAttention_v1(nn.Module):
    '''
    Setting our own parameters layers
    '''
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores/keys.shape[1]**0.5, dim=-1
            )
        
        context_vec = attn_weights @ values

        return context_vec

class SelfAttention_v2(nn.Module):
    '''
    Setting our own parameters layers
    '''
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores/keys.shape[1]**0.5, dim=-1
            )
        
        context_vec = attn_weights @ values

        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length, context_length),
                             diagonal=1)
                             )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values

        return context_vec
    

class MultiHeadAttentionWrapperClass(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

def create_dataload_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    '''
    The dataset class more or less encapsulates the sliding window logic. It breaks the text into chunks.
    '''
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader