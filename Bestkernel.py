#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
from cvxopt import matrix, spmatrix, solvers
import general
from kernels import kernel_normalize
from itertools import product
import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm as tqdm
solvers.options['show_progress'] = False

class nonlinear_kernel():

            #Implementation of Non-Linear Combination of kernel algorithm.
            #This implemented from the "Learning Non-Linear Combinations of Kernels"
            #This class returns the optimize weights of the non-linear combination of kernels
    
    def __init__(self, X, y, ID, kernels, C=1e-5, eps=1e-8, degree=2):

            #x parameter shows trainig features
            #y parameters shows trainig labels
            #ID shows ids
            #In kernel you will see the list of kernels
            #eps is decided to threshold calculatinf whether alpha is SV ro not
            #degree shows us the polinomial combinaion's order
            #c is a constant
        
        self.X = X
        self.y = y.loc[:, 'Bound']
        self.n = y.shape[0]
        self.ID = ID
        self.kernels = self.kernel_normalization(kernels)
        self.Id_X = np.array(X.loc[:, 'Id'])
        self.idx = np.array([np.where(self.ID == self.Id_X[i])[0] for i in range(len(self.Id_X))]).squeeze()
        self.kernels_fit = [K[self.idx][:, self.idx] for K in self.kernels]
        self.p = len(self.kernels_fit)
        self.C = C
        self.lbda = 1/(2*self.C*self.n)
        self.eps = eps
        self.degree = degree
        
    def numbers_step(self, ulti):
        r, o, z = np.arange(self.n), np.ones(self.n), np.zeros(self.n)
        K = np.sum((self.kernels_fit * ulti[:, None, None]), axis=0) ** self.degree
        P = matrix(K.astype(float), tc='d')
        q = matrix(-self.y, tc='d')
        G = spmatrix(np.r_[self.y, -self.y], np.r_[r, r + self.n], np.r_[r, r], tc='d')
        h = matrix(np.r_[o * self.C, z], tc='d')
        sol = solvers.qp(P, q, G, h)
        a = np.ravel(sol['x'])
        return a
        
    def kernel_get(self, u0=0, fnorm=1, n_iter=50, eta=1):
        ulti_s = self.fit(u0, fnorm, n_iter, eta)
        print('Alignment vector : ', ulti_s)
        Km = np.sum((self.kernels * ulti_s[:, None, None]), axis=0) ** self.degree
        print('Normalizing final kernel...')
        Km = kernel_normalize(Km)
        return Km


    def normalization(self, ulti, u0, fnorm):
        ulti_s = (ulti - u0)
        ulti_s_norm = ulti_s / np.sqrt(np.sum(ulti_s**2))
        ulti_s = ulti_s_norm * fnorm
        return ulti_s + u0
    
    
    def gradient(self, ulti, a1):
        K_t = np.sum(self.kernels_fit * ulti[:, None, None], axis=0) ** (self.degree - 1)
        gradient = np.zeros(self.p)
        for m in range(self.p):
            gradient[m] = a1.T.dot((K_t * self.kernels_fit[m])).dot(a1)
        return - self.degree * gradient
    

    def kernel_normalization(self, kernels):
        new_kernels = []
        for k, K in enumerate(kernels):
            print('Normalizing kernel {}...'.format(k + 1))
            new_kernels.append(kernel_normalize(K))
        return new_kernels


    def fit(self, u0=0, fnorm=10, n_iter=20, eta=1):
        u = np.ones(self.p)
        u = self.normalization(u, u0, fnorm)
        u = np.array([0 if u[i] < 0 else u[i] for i in range(self.p)])
        score_prev = np.inf
        for k in range(n_iter):
            print('Iteration {}, u={}, score={:0.5f}'.format(k, u, score_prev))
            a1 = self.numbers_step(u)
            g = self.gradient(u, a1)
            u_next = self.normalization(u - eta * g, u0, fnorm)
            u_next = np.array([0 if u_next[i] < 0 else u_next[i] for i in range(self.p)])
            score = np.linalg.norm(u_next - u, np.inf)
            if score > score_prev:
                eta *= 0.8
            if score < self.eps:
                return u_next
            u = u_next
            score_prev = score.copy()
        return u_next




def c_val(k, methods, Cs_NLK, Cs_SVM, degrees, lambdas):
    
        #We need to use cross_validation to make the weight optimize

        # First we have to Load the data
        
    d1, d2, d3, d4, kernels, ID = general.data_bring(methods)
    d_all = [d2, d3, d4]
    # Initialize results DataFrame
    p = len(kernels)
    n_param = len(Cs_NLK) * len(degrees) * len(lambdas)
    init = np.zeros(n_param)
    results = pd.DataFrame({'methods': [methods] * len(init), 'C NLCK': init, 'd': init, 'lambda': init, 'Best C CSVM': init, 'val acc': init})
    # Reformat
    X_train, y_train, X_val, y_val, X_test, kernels, ID = general.rechange_d1(d_all[k-1], kernels, ID)
    # Start cross validation on triplet (C, d, lambda)
    for i, param in tqdm(enumerate(product(Cs_NLK, degrees, lambdas)), total=n_param):
        C, d, lbda = param
        print('NLCK C={}, degree={}, lambda={}'.format(C, d, lbda))
        # it is time to kernel
        Km = NLCK(X_train, y_train, ID, kernels, C=C, eps=1e-9, degree=d).kernel_get(fnorm=lbda)
       
        C_opt, scores_tr, scores_te, mean_scores_tr, mean_scores_te =             general.c_val(Ps=Cs_SVM,
                                   d1=[X_train, y_train, X_val, y_val, X_test],
                                   algo='CSVM',
                                   kfolds=3,
                                   K=Km,
                                   ID=ID,
                                   pickleName='cv_C_SVM_NLCK_C{}_d{}_l{}_p{}_k{}.pkl'.format(C, d, lbda, p, k))
        # Save results
        results.iloc[i, 1:6] = C, d, lbda, C_opt, np.max(mean_scores_te)
    return results


# In[ ]:




