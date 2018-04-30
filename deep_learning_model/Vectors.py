import gensim
import os, string
import time

import numpy as np
import _pickle as pickle

import theano



def vectorize(text, vocab={}):
    #print("in vectorize")
    opts = [i for i in text.split(' ') if len(i)>0]

    cidx = 0
    tmp = []
    #iterate over the entire text, construct 3/2 grams first
    while cidx<len(opts):
        c0 = opts[cidx]
        if cidx+1<len(opts):
            c1 = opts[cidx+1]
        else:
            c1 = False
        if cidx+2<len(opts):
            c2 = opts[cidx+2]
        else:
            c2 = False
        if c2:
            s = c0+'_'+c1+'_'+c2
            if s in vocab:
                tmp.append(vocab[s])
                cidx+=3
                continue
        else:
            pass
        if c1:
            s = c0+'_'+c1
            if s in vocab:
                tmp.append(vocab[s])
                cidx+=2
                continue
        else:
            pass
        #no 3/2 grams, check the word
        if c0 in vocab:
            tmp.append(vocab[c0])
        elif c0.lower() in vocab:
            tmp.append(vocab[c0.lower()])
        else:
            #we have no token at this timestep, we could add a default?
            tmp.append(vocab['</s>'])
            pass
        cidx+=1
    return tmp



class GoogleVec(object):
    #logic for loading Google News vectors and transforming strings to word indices
    print("in GoogleVec class")
    def __init__(self, path='./GoogleNews-vectors-negative300.bin'):
        print("in init")
        self.path = path
        self.model = None
        self.vocab = {}


    def load(self):
        print("in load")
        t0 = time.time()
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.path,unicode_errors='ignore', binary=True)
        print (time.time() - t0)
        for i in range(len(self.model.index2word)):
            self.vocab[self.model.index2word[i]]=i


    def shared(self):
        print("in shared")
        #return theano.tensor.as_tensor_variable(self.model.syn0)
        return theano.shared(self.model.syn0)

    def transform(self, X, pad=0):
        #print("in transform")
        tmp = []
        maxlen=0
        for t in X:
            t = t.translate( string.punctuation)
            v = vectorize(t, vocab=self.vocab)
            if len(v)>maxlen:
                maxlen=len(v)
            tmp.append(v)
        e = np.zeros((len(X),maxlen)).astype('int32')
        e.fill(pad)

        for i in range(len(tmp)):
            v = tmp[i]
            e[i,:len(v)]=v
        return e




if __name__ == '__main__':

    gv = GoogleVec()
    gv.load()
    e = gv.shared()
    t = ['the quick brown fox jumped over the lazy dog','the. quick, brown! fox,, !']
    print (gv.transform(t))
    print (e)
