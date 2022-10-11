from datasets import Dataset
import pandas as pd
import random

from collections.abs import Sequence, Iterable, Hashable

class AutoAugmentDataset(Dataset):
    '''
    A dataset that uses the labels of the classes to construct the 
    (anchor, positive) pairs used in contrastive learning.
    Can also return the (anchor, positive, negative) pair.
    Returns a dict that is the pair with the required keys:
    { 
        'label': ..., 
        'anchor': ..., 
        'positive': ..., 
        'label_negative': ...,
        'negative': ...,  }
    '''

    def __init__(self, examples: Sequence, labels: Sequence[Hashable], return_negative: bool = True, return_labels: bool = True):
        """ Inits the dataset from a pandas dataframe, positives will be drawn
        from positives with the same anchor value, negatives will be drawn from 
        examples with other anchor values

        Args:
            examples (Sequence): the values
            anchors (Sequence): the example column labels
            return_negative (bool): whether to return negatives in the pair
            return_labels (bool): whether to return the label associated with the positive and negative examples            
        """

        assert not (return_negative == False and return_negative_anchor == True), "Return anchor is only permitted when returning negatives"
             
        self.examples = list(positives)
        self.labels = list(labels)

        self.return_negative = return_negative
        self.return_labels = return_labels

        # to speed up positive search collect indices in examples list
        # speeds up positive and negative sampling 
        self._positive_indices_for_label = defaultdict(list)
        for index, (positive, label) in enumerate(zip(self.examples, self.labels)):
            self._positive_indices_for_label[label].append(index)

    def __len__(self):
        return len(self.examples)

    def get_positive(self, anchor)
        '''Get positive for an anchor'''
        pos_idx = random.choice(self._positive_indices_for_label[anchor])
        return self.examples[pos_idx]

    def get_negative(self, anchor)
        '''Get a negative example for an anchor, also returns the negative_anchor'''
        # Rejection sampling for negative sampling
        neg_idx = random.randrange(0, len(self))
        while neg_idx in self.positive_indices_for_label[anchor]:
            neg_idx = random.randrange(0, len(self))
        
        return (self.labels[neg_idx], self.examples[neg_idx]) 

    def __getitem__(self, idx):
        anchor = self.examples[idx]

        result = {
            'anchor': anchor,
            'positive': self.get_positive(anchor)
        }
        
        if self.return_labels: result['label'] = self.labels[idx]

        if not return_negative:
            return result

        neg_label, neg_example = self.get_negative(anchor)

        result['negative'] = neg_example
        if self.return_labels: result['label_negative'] = neg_label

        return result

        
