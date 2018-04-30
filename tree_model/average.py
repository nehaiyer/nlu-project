import pandas as pd
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import _pickle
from score import *
from xgb_train_cvBodyId import fscore, perfect_score

def load_data():
    
    yuxi = pd.read_csv('predtest_cor2.csv', usecols=['Headline','Body ID','Stance','prob_0','prob_1','prob_2','prob_3'])
    print ('yuxi.shape:')
    print (yuxi.shape)
    

    doug = pd.read_csv('dosiblOutput.csv', usecols=['Headline','Body ID','Agree','Disagree','Discuss','Unrelated'])
    print ('doug.shape:')
    print (doug.shape)
    combine = pd.merge(yuxi, doug, on=['Headline', 'Body ID'], how='inner')
    print ('combine.shape:')
    print (combine.shape)

    targets = ['agree', 'disagree', 'discuss', 'unrelated']
    targets_dict = dict(zip(targets, range(len(targets))))
    combine['target'] = list(map(lambda x: targets_dict[x], combine['Stance']))
    y_meta = combine['target'].values
    x_meta = combine[['prob_0','prob_1','prob_2','prob_3','Agree','Disagree','Discuss','Unrelated']].values
    
    return x_meta, y_meta


def loadTest():
    
    yuxi = pd.read_csv('tree_pred_prob_cor2_new1.csv', usecols=['Headline','Body ID', 'prob_0','prob_1','prob_2','prob_3'],encoding = 'latin1')
    print ('yuxi.shape:')
    print (yuxi.shape)

    #doug = pd.read_csv('dosiblOutputFinal.csv', usecols=['Headline','Body ID','Agree','Disagree','Discuss','Unrelated'],encoding = 'latin1')  
    doug = pd.read_csv('pred_prob_.csv', usecols=['0','1','2','3'])  
    print ('doug.shape:')
    print (doug.shape)

    combine = pd.concat([yuxi, doug], axis=1)
    print ('combine.shape:')
    print (combine.shape)
    
    #x_meta = combine[['prob_0','prob_1','prob_2','prob_3','Agree','Disagree','Discuss','Unrelated']].values
    x_meta = combine[['prob_0','prob_1','prob_2','prob_3','0','1','2','3']].values
    return x_meta

def stack_test():
    
    param = {
        'w0': 1.0,
        'w1': 1.0
    }
    sumw = param['w0'] + param['w1']
    x_meta = loadTest()
    
    pred_agree = (x_meta[:,0]*param['w0'] + x_meta[:,4]*param['w1']) / sumw
    pred_disagree = (x_meta[:,1]*param['w0'] + x_meta[:,5]*param['w1']) / sumw
    pred_discuss = (x_meta[:,2]*param['w0'] + x_meta[:,6]*param['w1']) / sumw
    pred_unrelated = (x_meta[:,3]*param['w0'] + x_meta[:,7]*param['w1']) / sumw

    pred_y = np.hstack([pred_agree.reshape((-1,1)), pred_disagree.reshape((-1,1)), pred_discuss.reshape((-1,1)), pred_unrelated.reshape((-1,1))])
    print ('pred_agree.shape:')
    print (pred_agree.shape)
    print ('pred_disagree.shape:')
    print (pred_disagree.shape)
    print ('pred_discuss.shape:')
    print (pred_discuss.shape)
    print ('pred_unrelated.shape:')
    print (pred_unrelated.shape)

    print ('pred_y.shape:')
    print (pred_y.shape)
    pred_y_idx = np.argmax(pred_y, axis=1)
    predicted = [LABELS[int(a)] for a in pred_y_idx]
    
    #stances = pd.read_csv("test_stances_unlabeled_processed.csv")
    #df_output = pd.DataFrame()
    #df_output['Headline'] = stances['Headline']
    #df_output['Body ID'] = stances['Body ID']
    #df_output['Stance'] = predicted
    #df_output.to_csv('averaged_2models_cor4.csv', index=False)
    return pred_y_idx

def stack_cv(param):
    
    #x_meta, y_meta = load_data()
    sumw = param['w0'] + param['w1'] 
    pred_agree = (x_meta[:,0]*param['w0'] + x_meta[:,4]*param['w1']) / sumw
    pred_disagree = (x_meta[:,1]*param['w0'] + x_meta[:,5]*param['w1']) / sumw
    pred_discuss = (x_meta[:,2]*param['w0'] + x_meta[:,6]*param['w1']) / sumw
    pred_unrelated = (x_meta[:,3]*param['w0'] + x_meta[:,7]*param['w1']) / sumw

    pred_y = np.hstack([pred_agree.reshape((-1,1)), pred_disagree.reshape((-1,1)), pred_discuss.reshape((-1,1)), pred_unrelated.reshape((-1,1))])
    print ('pred_agree.shape:')
    print (pred_agree.shape)
    print ('pred_disagree.shape:')
    print (pred_disagree.shape)
    print ('pred_discuss.shape:')
    print (pred_discuss.shape)
    print ('pred_unrelated.shape:')
    print (pred_unrelated.shape)

    print ('pred_y.shape:')
    print (pred_y.shape)
    print ('y_meta.shape:')
    print (y_meta.shape)
    
    pred_y_label = np.argmax(pred_y, axis=1)
    predicted = [LABELS[int(a)] for a in pred_y_label]
    actual = [LABELS[int(a)] for a in y_meta]    

    score, _ = score_submission(actual, predicted)
    s_perf, _ = score_submission(actual, actual)

    cost = float(score) / s_perf

    #cost = log_loss(y_meta, pred_y, labels = [0, 1, 2, 3])
    
    return -1.0 * cost


def hyperopt_wrapper(param):
    
    print ("++++++++++++++++++++++++++++++")
    for k, v in sorted(param.items()):
        print ("%s: %s" % (k,v))

    loss = stack_cv(param)
    print ("-cost: ", loss)

    return {'loss': loss, 'status': STATUS_OK}

def run():

    param_space = {

            'w0': 1.0,
            'w1': hp.quniform('w1', 0.01, 2.0, 0.01),
            'max_evals': 800
            }
    
    
    trial_counter = 0
    trials = Trials()
    objective = lambda p: hyperopt_wrapper(p)
    best_params = fmin(objective, param_space, algo=tpe.suggest,\
        trials = trials, max_evals=param_space["max_evals"])
    
    print ('best parameters: ')
    for k, v in best_params.items():
        print ("%s: %s" % (k ,v))
    
    trial_loss = np.asarray(trials.losses(), dtype=float)
    best_loss = min(trial_loss)
    print ('best loss: ', best_loss)
# Computes competition score
def scorer(pred, truth):
    # Maximum possible score
    max_score = 0.25 * sum(truth == 3) + 1 * sum(truth != 3)
    # Computing achieved sore
    # Score from unrelated correct
    unrelated_score = 0.25 * sum((truth == 3) & (pred == truth))
    # Score from related correct, but specific class incorrect
    related_score1 = 0.25 * sum((truth != 3) & (pred != truth) & (pred != 3))
    # Score from getting related correct, specific class correct
    related_score2 = 0.75 * sum((truth != 3) & (pred == truth))

    final_score = (unrelated_score + related_score1 + related_score2) / max_score
    print(unrelated_score + related_score1 + related_score2)
    return final_score


if __name__ == '__main__':
    
    #x_meta, y_meta = load_data()
    #run()
    #loadTest()
    y_pred = stack_test()
    actual = pd.read_csv('competition_test_stances_new.csv')

    y = actual['Stance']
    predicted = pd.read_csv('averaged_2models_cor4_processed.csv')
    #y_pred = predicted['Stance']
    print("shapes:")
    print((len(y_pred),len(y)))
    final_score = scorer(y_pred,y)

    print(final_score)
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
    #stack_test()
