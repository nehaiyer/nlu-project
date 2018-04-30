from FeatureGenerator import *
import pandas as pd
import numpy as np
import _pickle
from nltk.tokenize import sent_tokenize
from helpers import *

import sys
sys.path.append('./mscproject/bin')
sys.path.append('./mscproject/src')
from run_calc_hungarian_alignment_score import *

class AlignmentFeatureGenerator(FeatureGenerator):

    '''
        compute the type-dependency graphs of headline and each sentence 
        of body text, find the best alignment
    '''

    def __init__(self, name='alignmentFeatureGenerator'):
        super(AlignmentFeatureGenerator, self).__init__(name)


    def process(self, df):
        
        n_train = df[~df['target'].isnull()].shape[0]
        n_test = df[df['target'].isnull()].shape[0]
        
        # only process test
        df = df[df['target'].isnull()]

        #df = df.iloc[:100, :]
        print ('generating word alignment features')
        #df['headline_clean'] = df['Headline'].map(lambda x: x.decode('utf-8').encode('ascii', errors='ignore'))
        #df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore')))
        df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        print ('sentences tokenized')
        #df['align_score_max'] = df.apply(lambda x: np.max([calc_hungarian_alignment_score(x['headline_clean'], bs, x.name)[1] for bs in x['body_sents']]), axis=1)
        df['align_score_max'] = df.apply(lambda x: np.max([calc_hungarian_alignment_score(x['Headline'], bs, x.name)[1] for bs in x['body_sents']]), axis=1)
        # return if the headline or matched body_sent is negated?
        # already taken of by the sentiment features?
        print ('word alignment features')
        print (df[['align_score_max', 'Stance']])
        
        # split into train, test portion and save them in separate files
        #train = df[~df['target'].isnull()]
        #xWalignTrain = train['align_score_max'].values
        #outfilename_walign_train = "train.walign.pkl"
        #with open(outfilename_walign_train, "wb") as outfile:
        #    _pickle.dump(xWalignTrain, outfile, -1)
        #print ('word alignment features for training saved in %s' % outfilename_walign_train)
        
        if n_test > 0:
            test = df[df['target'].isnull()]
            xWalignTest = test['align_score_max'].values
            outfilename_walign_test = "test.walign.pkl"
            with open(outfilename_walign_test, "wb") as outfile:
                _pickle.dump(xWalignTest, outfile, -1)
            print ('word alignment features for test saved in %s' % outfilename_walign_test)


    def read(self, header='train'):

        filename_walign = "%s.walign.pkl" % header
        with open(filename_walign, "rb") as infile:
            wordAlignScore = _pickle.load(infile).reshape(-1 ,1)
        
        #print (wordAlignScore
        print ('wordAlignScore.shape:')
        print (wordAlignScore.shape)

        return [wordAlignScore]

