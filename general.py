#!/usr/bin/env python
# coding: utf-8

# In[9]:


import import_ipynb
import pandas as pd
import numpy as np
import os
import warnings
import kernels as km
import pickle as pkl
import datetime
from tqdm import tqdm
from SVM import C_SVM
from Ridge import Kernelridge
import operator

path = 'C:\Mosig\Kernel Methodes\Lab' #'./Data'


# In[10]:


def train_data(k):
    
        #trainig data are loaded here 

    X, y = pd.read_csv('C:\Mosig\Kernel Methodes\Lab\Xtr' + str(k) + '.csv'), pd.read_csv('C:\Mosig\Kernel Methodes\Lab\Ytr' + str(k) + '.csv')
    y['Bound'] = y['Bound'].replace(0, -1)
    X.insert(1, 'k', k+1)
    y.insert(1, 'k', k+1)
    return X, y


def test_data(k):
    
        #testing data are loaded here

    X = pd.read_csv('C:\Mosig\Kernel Methodes\Lab\Xte' + str(k) + '.csv')
    X.insert(1, 'k', k+1)
    return X


# In[12]:


def split(X, y, p):
    
        #we split the data into 75% for training and the rest are for test

    idx_0, idx_1 = np.where(y.loc[:, "Bound"] == -1)[0], np.where(y.loc[:, "Bound"] == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    idx_tr0, idx_tr1 = idx_0[:int(p * n0)+1], idx_1[:int(p * n1)+1]
    idx_te0, idx_te1 = list(set(idx_0) - set(idx_tr0)), list(set(idx_1) - set(idx_tr1))
    idx_tr, idx_te = np.concatenate((idx_tr0, idx_tr1)), np.concatenate((idx_te0, idx_te1))
    X_train, y_train = X.iloc[idx_tr, :], y.iloc[idx_tr, :]
    X_val, y_val = X.iloc[idx_te, :], y.iloc[idx_te, :]
    return X_train, y_train, X_val, y_val


def spli_test():

    for k in range(3):
        X, y = train_data(k)
        if k == 0:
            X_train, y_train, X_val, y_val = split(X, y, 0.75)
            X_test = test_data(k)
        else:
            X_tr, y_tr, X_v, y_te = split(X, y, 0.75)
            X_train, X_val = pd.concat((X_train, X_tr), axis=0), pd.concat((X_val, X_v), axis=0)
            y_train, y_val = pd.concat((y_train, y_tr), axis=0), pd.concat((y_val, y_te), axis=0)
            X_test = pd.concat((X_test, test_data(k)), axis=0)
        X_train, X_val, y_train, y_val, X_test = resetIndex([X_train, X_val, y_train, y_val, X_test])
    return X_train, y_train, X_val, y_val, X_test


def selecting_k(k, X_train, y_train, X_val, y_val, X_test):
    
        #put all the training and testing data to 1 by selecting k

    idx_train = np.where(np.array(X_train.loc[:, 'k']) == k)[0]
    idx_val = np.where(np.array(X_val.loc[:, 'k']) == k)[0]
    idx_test = np.where(np.array(X_test.loc[:, 'k']) == k)[0]
    X_train_, y_train_ = X_train.iloc[idx_train], y_train.iloc[idx_train]
    X_val_, y_val_ = X_val.iloc[idx_val], y_val.iloc[idx_val]
    X_test_ = X_test.iloc[idx_test]
    return X_train_, y_train_, X_val_, y_val_, X_test_


# In[13]:


def import_data_to_train(method, all=True, replace=False):

        #preparing kernels in this section but first we had to Construct training and testing data

    file = 'training_data_'+method+'.pkl'
    if not all:
        X_train, y_train, X_val, y_val, X_test = spli_test()
        X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id'] + 1)
        X = pd.concat((X_train, X_val, X_test), axis=0)
        ID = X.loc[:, 'Id']
    else:
        if trainInRepo(file) and not replace:
            X_train, y_train, X_val, y_val, X_test, K, ID = pkl.load(open(os.path.join(path, file), 'rb'))
        else:
            X_train, y_train, X_val, y_val, X_test = spli_test()
            X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id']+1)
            X = pd.concat((X_train, X_val, X_test), axis=0)
            ID = np.array(X.loc[:, 'Id'])
            K = km.select_method(X, method)
            file = 'training_data_'+method+'.pkl'
            pkl.dump([X_train, y_train, X_val, y_val, X_test, K, ID], open(os.path.join(path, file), 'wb'))
    return X_train, y_train, X_val, y_val, X_test, K, ID


def loading(method):

        #Load kernel

    _, _, _, _, _, K, _ = import_data_to_train(method=method, replace=False)
    return K


def data_bring(methods):

    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, K, ID = import_data_to_train(method=methods[0], replace=False)
    X_train_1, y_train_1, X_val_1, y_val_1, X_test_1 = selecting_k(1, X_train, y_train, X_val, y_val, X_test)
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2 = selecting_k(2, X_train, y_train, X_val, y_val, X_test)
    X_train_3, y_train_3, X_val_3, y_val_3, X_test_3 = selecting_k(3, X_train, y_train, X_val, y_val, X_test)
    data = [X_train, y_train, X_val, y_val, X_test]
    data1 = [X_train_1, y_train_1, X_val_1, y_val_1, X_test_1]
    data2 = [X_train_2, y_train_2, X_val_2, y_val_2, X_test_2]
    data3 = [X_train_3, y_train_3, X_val_3, y_val_3, X_test_3]
    kernels = []
    for k, m in enumerate(methods):
        print('Kernel '+str(k+1)+'...')
        kernels.append(loading(m))
    if len(kernels) == 1:
        kernels = kernels[0]
    return data, data1, data2, data3, kernels, ID


# In[14]:


def cross_validation(Ps, data, algo, kfolds=5, **kwargs):
    
        #this part is implemented for algorithms that we used

    scores_tr = np.zeros((kfolds, len(Ps)))
    scores_te = np.zeros((kfolds, len(Ps)))
    X_tr, y_tr, X_te, y_te, _ = data
    X_train_ = pd.concat((X_tr, X_te)).reset_index(drop=True).sample(frac=1)
    y_train_ = pd.concat((y_tr, y_te)).reset_index(drop=True).iloc[X_train_.index]
    X_train_, y_train_ = X_train_.reset_index(drop=True), y_train_.reset_index(drop=True)
    n = X_train_.shape[0]
    p = int(n // kfolds)
    for k in tqdm(range(kfolds)):
        print('Fold {}'.format(k+1))
        q = p * (k + 1) + n % kfolds if k == kfolds - 1 else p * (k + 1)
        idx_val = np.arange(p * k, q)
        idx_train = np.setdiff1d(np.arange(n), idx_val)
        X_train, y_train = X_train_.iloc[idx_train, :], y_train_.iloc[idx_train, :]
        X_val, y_val = X_train_.iloc[idx_val, :], y_train_.iloc[idx_val, :]
        s_tr, s_te = [], []
        for P in Ps:
            if algo == 'CSVM':
                alg = C_SVM(C=P, print_callbacks=False, **kwargs)
            elif algo == 'Ridge':
                alg = Kernelridge(lbda=P, **kwargs)
            else:
                NotImplementedError('Please choose between "CSVM" or "Ridge"')
            alg.fit(X_train, y_train)
            pred_tr = alg.predict(X_train)
            score_tr = alg.score(pred_tr, y_train)
            pred_te = alg.predict(X_val)
            score_te = alg.score(pred_te, y_val)
            s_tr.append(score_tr)
            s_te.append(score_te)
            print('Constant={}, train_acc={:0.4f}, val_acc={:0.4f}'.format(P, score_tr, score_te))
        scores_tr[k], scores_te[k] = s_tr, s_te
    mean_scores_tr, mean_scores_te = np.mean(scores_tr, axis=0), np.mean(scores_te, axis=0)
    p_opt = Ps[np.argmax(mean_scores_te)]
    print('Best constant={}, val_acc={:0.4f}'.format(p_opt, np.max(mean_scores_te)))
    return p_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te


# In[15]:


def preict_send_to_output(algos, X_tests):

        #computing the prediction
        
    for k, alg in enumerate(algos):
        X_test = X_tests[k]
        pred_test = alg.predict(X_test).astype(int)
        if k == 0:
            y_test = pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})
        else:
            y_test = pd.concat((y_test, pd.DataFrame({'Id': X_test.Id, 'Bound': pred_test})))
    y_test.Id = np.arange(1000 * len(algos))
    y_test.Bound = y_test.Bound.replace(-1, 0)
    t = datetime.datetime.now().time()
    y_test.to_csv('y_test_'+'.csv', index=False)
    return y_test


# In[16]:


def rechange(data, kernels, ID):

        #rechange data in order to make computations faster for NLCkernel
        #However our runing spent more than 19 hours

    X_train, y_train, X_val, y_val, X_test = data
    ID_ = np.concatenate(
        (np.array(X_train.loc[:, 'Id']), np.array(X_val.loc[:, 'Id']), np.array(X_test.loc[:, 'Id'])))
    idx = np.array([np.where(ID == ID_[i])[0] for i in range(len(ID_))]).squeeze()
    kernels_ = []
    for K in tqdm(kernels):
        kernels_.append(K[idx][:, idx])
    ID_ = np.arange(ID_.shape[0])
    X_train.Id = ID_[:X_train.shape[0]]
    X_val.Id = ID_[X_train.shape[0]:(X_train.shape[0] + X_val.shape[0])]
    X_test.Id = ID_[(X_train.shape[0] + X_val.shape[0]):(
            X_train.shape[0] + X_val.shape[0] + X_test.shape[0])]
    y_train.Id = ID_[:y_train.shape[0]]
    y_val.Id = ID_[X_train.shape[0]:(X_train.shape[0] + X_val.shape[0])]
    return X_train, y_train, X_val, y_val, X_test, kernels_, ID_


def trainInRepo(file):

    return file in os.listdir(path)


def resetIndex(df):

    D = []
    for d in df:
        d.reset_index(drop=True, inplace=True)
        D.append(d)
    return D

