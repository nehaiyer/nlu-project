import ngram
#from pandas import DataFrame as pd
import pandas as pd
import numpy as np
import _pickle
from helpers import *
from CountFeatureGenerator import *
from TfidfFeatureGenerator import *
from SvdFeatureGenerator import *
from Word2VecFeatureGenerator import *
from SentimentFeatureGenerator import *
#from AlignmentFeatureGenerator import *

def process():

    read = True
    if not read:
    
        body_train = pd.read_csv("train_bodies_processed.csv")
        stances_train = pd.read_csv("train_stances_processed.csv", encoding='utf-8')
        # training set
        train = pd.merge(stances_train, body_train, how='left', on='Body ID')
        targets = ['agree', 'disagree', 'discuss', 'unrelated']
        targets_dict = dict(zip(targets, range(len(targets))))
        train['target'] = list(map(lambda x: targets_dict[x], train['Stance']))
        print (targets_dict)
        print ('train.shape:')
        print (train['target'])
        n_train = train.shape[0]

        data = train
        # read test set, no 'Stance' column in test set -> target = NULL
        # concatenate training and test set
        test_flag = True
        if test_flag:
            body_test = pd.read_csv("test_bodies_processed.csv")
            headline_test = pd.read_csv("test_stances_unlabeled.csv", encoding='utf-8')
            test = pd.merge(headline_test, body_test, how="left", on="Body ID")
            
            data = pd.concat((train, test)) # target = NaN for test set
            print (data)
            print ('data.shape:')
            print (data.shape)

            train = data[~data['target'].isnull()]
            print (train)
            print ('train.shape:')
            print (train.shape)
            
            test = data[data['target'].isnull()]
            print (test)
            print ('test.shape:')
            print (test.shape)

        #data = data.iloc[:100, :]
        
        #return 1
        
        print ("generate unigram")
        #data["Headline_unigram"] = list(data["Headline"].map(lambda x: preprocess_data(x)))
        print (data["Headline"])
        data["Headline_unigram"] =   list(map ( preprocess_data ,data["Headline"]))
        print (data["Headline_unigram"])
        data["articleBody_unigram"] = list(map( preprocess_data ,data["articleBody"]))

        print ("generate bigram")
        join_str = "_"
        data["Headline_bigram"] = list(map(lambda x: ngram.getBigram(x, join_str),data["Headline_unigram"]))
        data["articleBody_bigram"] =list( map(lambda x: ngram.getBigram(x, join_str),data["articleBody_unigram"]))
        
        print ("generate trigram")
        join_str = "_"
        data["Headline_trigram"] = list(map(lambda x: ngram.getTrigram(x, join_str),data["Headline_unigram"]))
        data["articleBody_trigram"] = list(map(lambda x: ngram.getTrigram(x, join_str),data["articleBody_unigram"]))
        
        with open('data.pkl', 'wb') as outfile:
            _pickle.dump(data, outfile, -1)
            print ('dataframe saved in data.pkl')

    else:
        with open('data.pkl', 'rb') as infile:
            data = _pickle.load(infile)
            print ('data loaded')
            print ('data.shape:')
            print (data.shape)
    #return 1

    # define feature generators
    countFG    = CountFeatureGenerator()
    tfidfFG    = TfidfFeatureGenerator()
    svdFG      = SvdFeatureGenerator()
    word2vecFG = Word2VecFeatureGenerator()
    sentiFG    = SentimentFeatureGenerator()
    #walignFG   = AlignmentFeatureGenerator()
    #generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    #generators = [ sentiFG]
    generators = [tfidfFG]
    #generators = [countFG]
    #generators = [walignFG]
    
    for g in generators:
        g.process(data)
    
    for g in generators:
        g.read('train')
    
    #for g in generators:
    #    g.read('test')

    print ('done')


if __name__ == "__main__":
    
    process()


