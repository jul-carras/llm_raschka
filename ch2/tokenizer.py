import re

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