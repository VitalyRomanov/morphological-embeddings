import os
from collections import Counter
import pickle

class Vocabulary:
    def __init__(self,voc_path):
        self.voc = load_voc(voc_path)
        self.inv_voc = make_inv_voc(self.voc)
        self.counts = load_counts(voc_path,self.voc)
        self.size = len(self.voc)

    def get_counts(self, word):
        w_id = self.voc.get(word, None)
        if w_id is None:
            raise Exception("word is not in vocabulary")

        return self.counts[w_id]

    def w2id(self, word):
        w_id = self.voc.get(word, None)
        if w_id is None:
            raise Exception("word is not in vocabulary")
        return w_id

    def id2w(self, id):
        w = self.inv_voc.get(id, None)
        if w is None:
            raise Exception("word is not in vocabulary")
        return w

    def most_common(self,N):
        mc = []
        for w_id, count in self.counts.most_common(N):
            mc.append((self.id2w(w_id), count))
        return mc

def make_inv_voc(vocab):
    return {id: word for word, id in vocab.items()}

def load_voc(path):
    vocab = dict()

    voc_path = os.path.join(path, "wiki_vocab")
    voc_path_pkl = os.path.join(path, "wiki_vocab.pkl")

    if os.path.isfile(voc_path_pkl):
        vocab = pickle.load(open(voc_path_pkl, "rb"))
    else:
        with open(voc_path, "r") as vocab_file:
            lines = vocab_file.read().split("\n")
            for line in lines:
                elem = line.split(" ")
                if len(elem) == 2:
                    vocab[elem[0]] = int(elem[1])
        pickle.dump(vocab, open(voc_path_pkl, "wb"), protocol=4)
    return vocab

def load_counts(path, vocab):
    token_counter = Counter()

    counts_path = os.path.join(path, "wiki_tokens")
    counts_path_pkl = os.path.join(path, "wiki_tokens.pkl")

    if os.path.isfile(counts_path_pkl):
        token_counter = pickle.load(open(counts_path_pkl, "rb"))
    else:
        with open(counts_path, "r") as token_file:
            lines = token_file.read().split("\n")
            for line in lines:
                elem = line.split(" ")
                if len(elem) == 2:
                    token_counter[vocab[elem[0]]] = int(elem[1])
        pickle.dump(token_counter, open(counts_path_pkl, "wb"), protocol=4)
    return token_counter
