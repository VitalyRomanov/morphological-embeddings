from WikiLoader import WikiDataLoader
from Tokenizer import Tokenizer
from Vocabulary import Vocabulary
from BayesSkipGram import model_fast as model
# import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pprint import pprint

from collections import Counter
import sys
import numpy as np
import pickle


BATCH_SIZE=1000
CONTEXT_CAPACITY=4
ACTIVE_VOC_SIZE=10000
EMB_SIZE = 200
EPOCHS = 10
WIKI_PATH = "/Users/LTV/dev/EnWiki"

if CONTEXT_CAPACITY % 2 != 0:
        raise ValueError("Context length should be even")


class DataFeeder:
    """
    Implements data feeder interface. Requires vocabulary with method
    getId. Provides interface for returning next context. Trims context
    with new line character.
    """
    def __init__(self, voc, source="wiki", path=""):
        if source != "wiki":
            raise NotImplementedError
        
        if source == "wiki" and path == "":
            raise ValueError("Provide path for wiki location")

        self.CONT_W = CONTEXT_CAPACITY + 1
        self.source = WikiDataLoader(path, jsonf=True)
        self.tok = Tokenizer()

        self.cDoc = None

    def nextDoc(self):

        nDoc = self.source.next_doc()

        if nDoc is None:
            return None
        
        self.cDoc = nDoc.split("\n")

    def nextLine(self):

        if not self.cDoc:
            self.nextDoc()
        if self.cDoc:
            return self.tok(self.cDoc.pop(0))
        else:
            return None

df = DataFeeder("voc", path=WIKI_PATH)  

line = df.nextLine()
while line is not None:
    while line == "":
        line = df.nextLine()
    line = df.nextLine()
sys.exit()


def tsne(savepath, embs, voc):
    # sample = np.random.randint(embs.shape[0],size=5000)
    sample = np.arange(ACTIVE_VOC_SIZE)
    new_values = TSNE(n_components=2).fit_transform(
                                            embs[sample])

    labels = [voc.inv_ids[s] for s in sample]

    x = []; y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(savepath)
    plt.show()
    # plt.close()

def prep_dataset():
    wiki_path = WIKI_PATH

    if CONTEXT_CAPACITY % 2 != 0:
        raise Exception("Context length should be even")

    context_window = CONTEXT_CAPACITY + 1

    print("Loading...",end = "")
    wiki = WikiDataLoader(wiki_path)
    voc = Vocabulary()
    tok = Tokenizer()
    print("done")

    wiki_doc = wiki.next_doc()
    wikiprep = open("WikiPrepData.txt","w")

    i = 0
    while wiki_doc:
        doc = tok(wiki_doc)
        voc.add(doc)

        sample = np.array(voc.text2ids(doc))
        indexer = np.arange(context_window)[None, :] + np.arange(len(sample)-context_window)[:, None]

        smpl = sample[indexer]

        for row in smpl:
            for val in row:
                wikiprep.write("%d " % val)
            wikiprep.write("\n")

        i+= 1
        if i == 2000:
            break
        wiki_doc = wiki.next_doc()

    pickle.dump(voc, open("WikiPrepVoc.pkl", "wb"))
    print("Vocabulary ready")


# prep_dataset()
voc = pickle.load(open("WikiPrepVoc.pkl", "rb"))
model = model(emb_size = EMB_SIZE, context_p_negative = 2*CONTEXT_CAPACITY+1, voc_size = ACTIVE_VOC_SIZE, miniBATCH_SIZE=BATCH_SIZE)

# sys.exit()

class NegStreamer:
    def __init__(self, voc):
        self.voc = voc
        self.temp = []
        self.pos = 0

    def get(self, dim1, dim2=1, limit_top=-1):
        nsamples = dim1*dim2
        if self.pos + nsamples > len(self.temp):
            self.temp = voc.sample(nsamples*2, limit_top=limit_top)
            self.pos = 0

        send = self.temp[self.pos: self.pos + nsamples]
        self.pos += nsamples
        return np.array(send).reshape(dim1, dim2)

class PrepDataReader:

    def __init__(self, filepath, vocabulary):
        self.doc_path = filepath

    def open(self):
        self.doc = open(self.doc_path, "r")

    def reset(self):
        self.doc.close()
        self.open()

    def next_batch(self, BATCH_SIZE, clip_ids=-1):
        try:
            vals = " ".join(self.doc.readline().strip() for b in range(BATCH_SIZE))
            batch = np.fromstring(vals, dtype=int, sep=' ').reshape(BATCH_SIZE, CONTEXT_CAPACITY + 1)
            if clip_ids != -1:
                batch[batch >= clip_ids] = 0
        except:
            batch = None
        return batch




neg_stream = NegStreamer(voc)
data = PrepDataReader("WikiPrepData.txt", voc)
data.open()

i=0
with tf.Session() as sess:
    # ########### plot only
    # model['saver'].restore(sess, "/Volumes/Seagate 2nd part/lang/model_final.ckpt")
    # embs = sess.run(model['emb'])
    # tsne("/Volumes/Seagate 2nd part/lang/1.svg" , embs, voc)
    # ###########

    sess.run(model['init'])



    try:
        for e in range(EPOCHS):
            data.reset()

            batch = data.next_batch(BATCH_SIZE=BATCH_SIZE, clip_ids=ACTIVE_VOC_SIZE)
            while batch is not None:
                i += 1
                # bring central word to the front
                contral_word_pos = int((CONTEXT_CAPACITY)/2)

                indexer = np.concatenate([np.array([contral_word_pos]), np.arange(contral_word_pos), np.arange(contral_word_pos+1, CONTEXT_CAPACITY+1)])
                b = np.concatenate([batch[:,indexer], neg_stream.get(BATCH_SIZE,CONTEXT_CAPACITY,limit_top=ACTIVE_VOC_SIZE)], axis=1)
                _, l = sess.run([model['train'], model['loss']], {model['wids']:b})
                print(l)
                # if i % 1000 == 0:
                #     print(l)
                line = data.next_batch(BATCH_SIZE=BATCH_SIZE, clip_ids=ACTIVE_VOC_SIZE)
                # break


            model['saver'].save(sess, "./res/model_%d.ckpt" % e)
            embs = sess.run(model['emb'])
            tsne("./bsg/%d.svg" % e, embs, voc)
    except KeyboardInterrupt:
        print("wtf")

        model['saver'].save(sess, "./res/model_final.ckpt")
        embs = sess.run(model['emb'])
        tsne("./res/%d.svg" % e, embs, voc)



# wikiprep = open("WikiPrepData.txt","w")
# wiki = WikiDataLoader(wiki_path)
# wiki_doc = wiki.next_doc()
# i = 0
# while wiki_doc:
#     doc = tok(wiki_doc)
#
#
#
#     # smpl = np.concatenate([sample[indexer], np.array(voc.sample(len(sample)))[indexer][:,:-1]], axis=1)
#     # print(smpl)
#
#
#     # for ii in range(len(doc) - 5):
#     #     sample =
#     #     print("\t",ii)
#     #     sample = voc.text2ids(doc[ii:ii+5])
#     #     neg_sample = voc.sample(4)
#     #     sample.extend(neg_sample)
#
#     i+= 1
#     if i % 200 == 0:
#         print(i)
#     if i == 2000:
#         break
#
#     wiki_doc = wiki.next_doc()
#
# # for a in voc.most_common():
# #     print(a)
