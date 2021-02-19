#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
import numpy as np
from tqdm import tqdm as tqdm
from itertools import product, combinations
from copy import deepcopy
from scipy.sparse.linalg import eigs
from numpy.linalg import multi_dot


# In[2]:


def u_phi(x, k, b_s):

        #We started with spectrum kernel and here we compute the feature vector of x sequence.
        # In b_s we have a list of all combination of 'A', 'C', 'G', 'T'
   
    u_phi = np.zeros(len(b_s))
    for i in range(len(x) - k + 1):
        seq = x[i:i + k]
        for i, b in enumerate(b_s):
            u_phi[i] += (b == seq)
    return u_phi


def compute_spectrum_k(X, k):

        #For every input sequences we compute this one for Spectrum kernel
        #Xshows some features
        #K show s the length of the sequence

    n = X.shape[0]
    K = np.zeros((n, n))
    b_s = [''.join(c) for c in product('ACGT', repeat=k)]
    u_phi = []
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Computing feature vectors'):
        u_phi.append(u_phi(x, k, b_s))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(u_phi[i], u_phi[j])
                K[j, i] = K[i, j]
    K = K
    return K


# In[3]:


def weight_b(d, k):

        #For this step we need to calculate the weights for Weighted degree kernel(k)
        #d shows the maximum degree

    return 2 * (d - k + 1) / d / (d + 1)


def com_two_seq(x, y, d, L):

    c_t = 0
    for k in range(1, d + 1):
        weight_b_k = weight_b(d, k)
        c_st = 0
        for l in range(1, L - k + 1):
            c_st += (x[l:l + k] == y[l:l + k])
        c_t += weight_b_k * c_st
    return c_t


def com_k_seq(X, d):

        #weighted degree kernel(d) needs to compute k for each sequences
        #d shows maximom degree like preivious

    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        L = len(x)
        K[i, i] = L - 1 + (1 - d) / 3
        for j, y in enumerate(X.loc[:, 'seq']):
            if j > i:
                K[i, j] = com_two_seq(x, y, d, L)
                K[j, i] = K[i, j]
    return K


# In[4]:


def mega(s):
   
        #Compute mega coefficients for Weight Degree Kernel with Shifts
    
        #return: maga(s)
   
    return 1/2/(s+1)


def com_k_x_y(x, y, d, S, L):

    c_t = 0
    for k in range(1, d + 1):
        weight_b_k = weight_b(d, k)
        c_st = 0
        for i in range(1, L - k + 1):
            for s in range(0, S+1):
                if s+i < L:
                    c_st += mega(s) * ((x[i+s:i+s+k] == y[i:i+k]) + (x[i:i+k] == y[i+s:i+s+k]))
        c_t += weight_b_k * c_st
    return c_t


def com_shift_k(X, d, S):

    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        L = len(x)
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = com_k_x_y(x, y, d, S, L)
                K[j, i] = K[i, j]
    return K


# In[5]:


def fvector_MK(x, k, m, b_s):

    #Compute feature vector of sequence x for Mismatch Kernel (k,m)

    phi_km = np.zeros(len(b_s))
    for i in range(101 - k + 1):
        seq = x[i:i + k]
        for i, b in enumerate(b_s):
            phi_km[i] += (np.sum(seq != b) <= m)
    return phi_km


def replace_let_num(x):

    return x.replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4')


def format(x):
          #format the data
        
    return np.array(list(replace_let_num(x))).astype(int)


def mismatch_kernel(X, k, m):

    n = X.shape[0]
    K = np.zeros((n, n))
    b_s = np.array([format(''.join(c)) for c in product('ACGT', repeat=k)])
    phi_km_x = np.zeros((n, len(b_s)))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Computing feature vectors'):
        x = format(x)
        phi_km_x[i] = fvector_MK(x, k, m, b_s)
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_km_x[i], phi_km_x[j])
                K[j, i] = K[i, j]
    K = kernel_normalize(K)
    return K


# In[6]:


S = np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, 2], [0, -1, -2, 5]])


def sw_kernel(x, y, e=11, d=1, beta=0.5):

        #smith waterman kernel implementation

    x, y = format(x) - 1, format(y) - 1
    n_x, n_y = len(x), len(y)
    M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))] * 5
    for i in range(1, n_x):
        for j in range(1, n_y):
            M[i, j] = np.exp(beta * S[x[i], y[j]]) * max(1, X[i - 1, j - 1], Y[i - 1, j - 1], M[i - 1, j - 1])
            X[i, j] = max(np.exp(beta * d) * M[i - 1, j], np.exp(beta * e) * X[i - 1, j])
            Y[i, j] = max(np.exp(beta * d) * M[i, j - 1], np.exp(beta * d) * X[i, j - 1], np.exp(beta * e) * Y[i, j - 1])
            X2[i, j] = max(M[i - 1, j], X2[i - 1, j])
            Y2[i, j] = max(M[i, j - 1], X2[i, j - 1], Y2[i, j - 1])
    return (1/beta) * np.log(max(1, X2[n_x, n_y], Y2[n_x, n_y], M[n_x, n_y]))


def local_align_kernel(x, y, e, d, beta):

        #local alignment kernel implemented here

    x, y = format(x)-1, format(y)-1
    n_x, n_y = len(x), len(y)
    M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))]*5
    for i in range(1, n_x):
        for j in range(1, n_y):
            M[i, j] = np.exp(beta * S[x[i], y[j]]) * (1 + X[i-1, j-1] + Y[i-1, j-1] + M[i-1, j-1])
            X[i, j] = np.exp(beta * d) * M[i-1, j] + np.exp(beta * e) * X[i-1, j]
            Y[i, j] = np.exp(beta * d) * (M[i, j-1] + X[i, j-1]) + np.exp(beta * e) * Y[i, j-1]
            X2[i, j] = M[i-1, j] + X2[i-1, j]
            Y2[i, j] = M[i, j-1] + X2[i, j-1] + Y2[i, j-1]
    return (1/beta) * np.log(1 + X2[n_x, n_y] + Y2[n_x, n_y] + M[n_x, n_y])


def execute_al(X, e=11, d=1, beta=0.5, smith=0, eig=1):

    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = sw_kernel(x, y, e, d, beta) if smith else local_align_kernel(x, y, e, d, beta)
                K[j, i] = K[i, j]
    K1 = deepcopy(K)
    if eig == 1:
        vp = np.min(np.real(eigs(K1)[0]))
        s = vp if vp < 0 else 0
        np.fill_diagonal(K1, np.diag(K1) - s * np.ones(n))
    else:
        for i in tqdm(range(K1.shape[0]), desc='Empirical kernel'):
            for j in range(i, n):
                K1[i, j] = np.dot(K[i], K[j])
                K1[j, i] = K1[i, j]
    return K


# In[7]:


def mem_rec(func):
    
        #A Method to recursion memory

    memory = {}

    def mem_recd(*args):
        key = '-'.join('[%s]' % arg for arg in args)
        if key not in memory:
            memory[key] = func(*args)
        return memory[key]
    return mem_recd


@mem_rec
def rec_bk(lbda, k, x, y):

    if k == 0:
        return 1
    n_x, n_y = len(x), len(y)
    if n_x < k or n_y < k:
        return 0
    sub_x, sub_y = x[:-1], y[:-1]
    return (
            lbda * rec_bk(lbda, k, sub_x, y)
            + lbda * rec_bk(lbda, k, x, sub_y)
            - (lbda**2) * rec_bk(lbda, k, sub_x, sub_y)
            + ((lbda**2) * rec_bk(lbda, k-1, sub_x, sub_y) if x[-1] == y[-1] else 0)
           )

@mem_rec
def rec_kk(lbda, k, x, y):

    if k == 0:
        return 1
    n_x, n_y = len(x), len(y)
    if n_x < k or n_y < k:
        return 0
    sub_x = x[:-1]
    a = x[-1]
    return (
            rec_kk(lbda, k, sub_x, y)
            + (lbda**2) * sum(rec_bk(lbda, k-1, sub_x, y[:j]) for j in range(n_y) if y[j] == a)
           )


def compute_string_kernel(X, lbda, k):

    n = X.shape[0]
    K = np.zeros((n, n))
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = rec_kk(lbda, k, x, y)
                K[j, i] = K[i, j]
    return K


# In[8]:


def center_K(K):
 
        #Implemetation of center Kernel
    
    n = K.shape[0]
    B = np.eye(n) - np.ones((n, n))/n
    return multi_dot([B, K, B])


def kernel_normalize(K):

    if K[0, 0] == 1:
        print('Kernel already normalized')
    else:
        n = K.shape[0]
        diag = np.sqrt(np.diag(K))
        for i in range(n):
            d = diag[i]
            for j in range(i+1, n):
                K[i, j] /= (d * diag[j])
                K[j, i] = K[i, j]
        np.fill_diagonal(K, np.ones(n))
    return K


# In[9]:


def kernel_gappy(x, k, g, b_s):

    phi = np.zeros(len(b_s))
    gap_set = sum([list(combinations(x[i:i+k], k-g)) for i in range(101 - k + 1)], [])
    for i, b in enumerate(b_s):
        phi[i] = (b in gap_set)
    return phi


def compute_gappy_k(X, k, g):

    n = X.shape[0]
    K = np.zeros((n, n))
    b_s = np.array([format(''.join(c)) for c in product('ACGT', repeat=k)])
    for i, x in tqdm(enumerate(X.loc[:, 'seq']), total=n, desc='Building kernel'):
        x = format(x)
        phi_x = kernel_gappy(x, k, g, b_s)
        for j, y in enumerate(X.loc[:, 'seq']):
            if j >= i:
                K[i, j] = np.dot(phi_x, kernel_gappy(y, k, g, b_s))
                K[j, i] = K[i, j]
    K = kernel_normalize(K)
    return K


# In[11]:


def select_method(X, method):

    m = method.split('_')
    if method[:2] == 'spectrum_kernel':
        k = int(m[1][1:])
        K = compute_spectrum_k(X, k)
    elif method[:2] == 'weighted_degree' and method[2] != 'S':
        print(m)
        d = int(m[1][1:])
        K = com_k_seq(X, d)
    elif method[:2] == 'mismatch_lernel':
        k, m = int(m[1][1:]), int(m[2][1:])
        K = mismatch_kernel(X, k, m)
    elif method[:2] == 'local_align_kernel':
        e, d, beta = [float(m[i][1:]) for i in range(1, 4)]
        smith, eig = int(m[4][5:]), int(m[5][3:])
        K = execute_al(X, e, d, beta, smith, eig)
    elif method[:3] == 'weighted_degree_shift':
        d, S = int(m[1][1:]), int(m[2][1:])
        K = com_shift_k(X, d, S)
    elif method[:2] == 'string_kernel':
        lbda, k = float(m[1][1:]), int(m[2][1:])
        K = compute_string_kernel(X, lbda, k)
    elif method[:2] == 'gappy_kernel':
        k, g = int(m[1][1:]), int(m[2][1:])
        K = get_kernel_gappy(X, k, g)
    else:
        NotImplementedError('Method not implemented. Please refer to the documentation for choosing among available methods')
    return K

