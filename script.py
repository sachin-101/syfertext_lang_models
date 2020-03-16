import spacy
import pickle
import mmh3
import os

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
    vectors = {}
    key2rows = {}
    words = {}
    for index, word in enumerate(nlp.vocab.strings):
        print(index, word)
        vector = nlp(word).vector
        key = hash_string(word)
        key2rows[key] =  index
        vectors[index] =  vector
        words[index] = word

    save_file(vectors, filename='vectors')
    save_file(key2rows, filename='key2rows')
    save_file(words, filename='words')
    

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    create_lang_model(nlp)

















































## So as of now, I can work with 3 of PRs. NIIIIICEEEEEEEEEEE.
## It's not that I can resolve any one of them, in just a short
## span of time, but I can resolve all of them, if I give enough time 
## to all.

## And, regarding force delete, the main feature he requires is to 
## add force delete in TensorPointers rather than, VirtualWorkers, cause
## they already seem to have one of those mehtods.