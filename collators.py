import functools
import torch
from torchvision.io import read_image, ImageReadMode

class TextCollator:
    def __init__(self, tokenizer, max_length=32, vocab=None):
        self.tokenizer = tokenizer
        self.max_len = max_length

        self.vocab = vocab

        self.tokenize_fun = functools.partial(self.tokenizer, 
            truncation=True, 
            padding='max_length', max_length=self.max_len,
            return_tensors='pt'
        )

    def __call__(self, examples):
        anchors = [example['anchor'] for example in examples]
        positives = [example['positive'] for example in examples]
        negatives = [example['negative'] for example in examples]
        
        tokenized_dict = {
            "anchor": self.tokenize_fun(anchors),
            "positive": self.tokenize_fun(positives),
            "negative": self.tokenize_fun(negatives),
        }

        if "label" in examples[0]:
            labels = [example['label'] for example in examples]
            labels_negative = [example['label_negative'] for example in examples]
            if self.vocab:
                # Get the numeric labels from the vocab if a vocab is passed
                return dict(
                    labels=torch.tensor([self.vocab[lbl] for lbl in labels], dtype=torch.long), 
                    labels_negative=torch.tensor([self.vocab[lbl] for lbl in labels_negative], dtype=torch.long),
                    **tokenized_dict
                )
            else:
                # Pass the labels as they are
                return dict(
                    labels=torch.tensor([lbl for lbl in labels], dtype=torch.long), 
                    labels_negative=torch.tensor([lbl for lbl in labels_negative], dtype=torch.long),
                    **tokenized_dict
                )
        return tokenized_dict    

class ImageCollator:
    def __init__(self, train_transform, vocab=None):
        self.train_transform = train_transform
        self.vocab = vocab

    def read_image(self, path: str):
        return read_image(path, ImageReadMode.RGB)
        
    def __call__(self, examples):
        anchors = torch.stack([self.train_transform(self.read_image(example['anchor'])) for example in examples])
        positives = torch.stack([self.train_transform(self.read_image(example['positive'])) for example in examples])
        negatives = torch.stack([self.train_transform(self.read_image(example['negative'])) for example in examples])

        result {
            "label": labels,
            "anchor": {'pixel_values': anchors},
            "positive": {'pixel_values': positives},
            "negative": {'pixel_values': negatives},
        }

        if 'label' in examples:
            labels = [examples['label'] for example in examples]
            if vocab:
                result['labels'] = torch.tensor([vocab[lbl] for lbl in labels], dtype=torch.long)
            else:
                result['labels'] = torch.tensor([lbl for lbl in labels], dtype=torch.long)
        
        return result