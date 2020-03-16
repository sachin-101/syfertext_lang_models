
import spacy
import pickle
import mmh3
import os
import numpy as np

# This script does the following:
# 1. Loads a spacy language model
# 2. Creates a language model for SyferText, with the following contents:
#     a. vectors (binary file) - dictionary{keys:index, values: vectors}
#     b. key2rows (binary file) - dictionary {keys: as hashes of token texts,
#                         values: index in 'vectors' of the corresponding vector} 
#     c. words (binary file) - list of strings for the StringObject of the spacy language model

def hash_string(string):
    key = mmh3.hash64(string, signed=False, seed=1)[0]
    return key

def save_file(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def create_lang_model(nlp):
    vocab_words = list(nlp.vocab.strings)[1:]  # Avoiding the first empty char
    vocab_size = len(vocab_words)
    embed_dim = nlp(vocab_words[0]).vector.shape[0]
    vectors = np.zeros((vocab_size, embed_dim))
    key2row = {}
    words = []
    for index, word in enumerate(vocab_words):
        print(index, word)
        vector = nlp(word).vector
        key = hash_string(word)
        key2row[key] =  index
        vectors[index] =  vector
        words.append(word) 

    save_file(vectors, filename='vectors')
    save_file(key2row, filename='key2row')
    save_file(words, filename='words')
    

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    create_lang_model(nlp)

