import numpy as np
from common import *

symbol2vec_map = {}

def fill_embeddings_map(alphabet):
    for i, char in enumerate(alphabet):
        hot_vector = np.zeros((len(alphabet), 1))
        hot_vector[i] = 1
        symbol2vec_map[char] = hot_vector

def get_embedded_sentence(snippet):
    embedded_snippet = np.empty((ALPHABET_SIZE, 0))
    for i in range(len(snippet)):
        if (i >= TELEGRAM_MESSAGE_MAX_LEN):
            break
        if (snippet[i] in symbol2vec_map):
            embedded_snippet = np.append(embedded_snippet, symbol2vec_map[snippet[i]], axis=1)
    return embedded_snippet