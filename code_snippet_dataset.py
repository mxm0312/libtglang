import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from numpy import nan
import numpy as np

from common import *
from word_embeddings import *

class CodeSnippetDataset(Dataset):
    """Dataset that contains code snippets"""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with snippets
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.sentence_embeddings_map = {}

        # read csv
        
        df = pd.read_csv(csv_file)
        df.drop(labels=['Unnamed: 0'], axis="columns", inplace=True)
        df = df.dropna()
        
        data = df.to_numpy()

        indxs = []
        count = 0
        for i, item in enumerate(data):
            if count == 2000:
                break
            if data[i][1] == 'SCALA':
                indxs.append(i)
                count+=1
        data = np.delete(data, indxs, axis=0)

        # Encode targets
        self.target = data[:, -1]
 
        for i, target in enumerate(self.target):      
            self.target[i] = targets_desctiption.index(target)
        self.data = data[:, 0]
        
        print(f'Dataset len: {data.shape[0]}')
        
       
        df = pd.DataFrame(data, columns = ['Snippet','language'])
        df["language"].value_counts().plot(kind='bar',figsize = (21,10), fontsize=20, rot=75)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedded_sentence = ''
        
        if idx in self.sentence_embeddings_map:
            embedded_sentence = self.sentence_embeddings_map[idx]
        else:
            embedded_sentence = get_embedded_sentence(self.data[idx])
            self.sentence_embeddings_map[idx] = embedded_sentence
            
        # Code snippet augmentations
        #start = random.randint(0, 896)
        #end = random.randint(960, 1024)
        
        # embedded_sentence = embedded_sentence[:, start:end]
        
        if (embedded_sentence.shape[1] != TELEGRAM_MESSAGE_MAX_LEN):
            diff = TELEGRAM_MESSAGE_MAX_LEN - embedded_sentence.shape[1]
            embedded_sentence = np.append(embedded_sentence, np.zeros((ALPHABET_SIZE, diff)), axis=1)
    
        target = np.zeros((CLASSES_NUM,))
        target[self.target[idx]] = 1.0
        
        torch_embedded_sentence = torch.tensor(embedded_sentence, dtype=torch.float32)
        torch_target = torch.tensor(target, dtype=torch.float32)
        
        return torch_embedded_sentence, torch_target