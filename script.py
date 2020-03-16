import spacy
import pickle
import mmh3
import os
import numpy as np

# This script creates a language model for SyferText, with the following files:
#     a. vectors - numpy array with dimension (vocab_size, embed_dim)
#     b. key2row - dictionary {keys: hashes of token texts,
#                         values: index in 'vectors' of the corresponding vector} 
#     c. words - list of strings from the StringObject of the spacy language model

def hash_string(string):
    key = mmh3.hash64(string, signed=False, seed=1)[0]
    return key

def save_file(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def create_lang_model(nlp):
    vocab = list(nlp.vocab.strings)     
    vocab_size = len(vocab)              
    embed_dim = nlp(vocab[0]).vector.shape[0]
    vectors = np.zeros((vocab_size, embed_dim)) 
    key2row = {}    
    words = []      
    for index, word in enumerate(vocab):
        vector = nlp(word).vector
        key = hash_string(word)
        key2row[key] =  index   # key and row of corresponding vector
        vectors[index] =  vector   
        words.append(word)

    save_file(vectors, filename='vectors')
    save_file(key2row, filename='key2row')
    save_file(words, filename='words')
    

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    create_lang_model(nlp)

