import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
from functools import partial

# %matplotlib inline
plt.rcParams['savefig.dpi'] = 100

#------------Projected subgradient Descent--------------
def matrix_psub(X, M, O1, O0, t, power = -1.0):
    U, s, V = la.svd(X, full_matrices = False)
    Z = np.dot(U, V)
        
    eta = np.power(t+1, power)
    
    X = X - eta * Z
        
    X = X * O0 + M * O1
        
    return X, np.sum(s)

def matrix_descent(update, M, O, M_true, I_0, T=int(1e4)):
    # calculating error is different from matrix_descent
    np.random.seed(0)
    X = np.random.rand(M.shape[0], M.shape[1])
    O1 = (O == 1)
    O0 = (O == 0)
               
    error = []
    nuclear_norms = []
        
    for t in xrange(T):
        # update A (either subgradient or frank-wolfe)
        X, rank = update(X, M, O1, O0, t)
                
        # record error and l1 norm
        if (t % 1 == 0) or (t == T - 1):
            if len(I_0):
                U, s, V = la.svd(X[:,0:I_0[0]], full_matrices = False)
                error.append(la.norm((X - M_true)[:,0:I_0[0]], 'fro') / np.size(X[:,0:I_0[0]]))
            else: # I_0 is empty
                U, s, V = la.svd(X, full_matrices = False)
                error.append(la.norm(X - M_true, 'fro') / np.size(X))
            nuclear_norms.append(np.sum(s))
            
            assert not np.isnan(error[-1])

    return X, error, nuclear_norms

#--------------FPC---------------------
def fp_iter(X, M, O1, mu, tau = 1.0):
    Y = X - tau * (X - M) * O1;
    U, s, V = la.svd(Y, full_matrices = False)
    shift_s = s - tau * mu
    shrink_s = (shift_s > 0) * shift_s
    X = np.dot(np.multiply(U, shrink_s), V)
    return X

def fp_loop(X, M, O1, mu, M_true, I_0, xtol = 1e-10, max_iter = 200):
    errors = []
    ranks = []
    for k in range(max_iter):
#         print k, la.norm(X, 'fro')
        X_norm = max(1.0, la.norm(X, 'fro'))
#         print X_norm
        if len(I_0):
            U, s, V = la.svd(X[:,0:I_0[0]], full_matrices = False)
            errors.append(la.norm((X - M_true)[:,0:I_0[0]], 'fro') / np.size(X[:,0:I_0[0]]))
        else: # I_0 is empty
            U, s, V = la.svd(X, full_matrices = False)
            errors.append(la.norm(X - M_true, 'fro') / np.size(X))
        ranks.append(np.sum(s))
        X_next = fp_iter(X, M, O1, mu)
        diff_norm = la.norm(X_next - X, 'fro')
        if diff_norm/X_norm < xtol:
            break
        X = X_next
    return X, errors, ranks

def fpc(M, O1, M_true, I_0,mu_final = 1e-4, eta = 0.25):
    mu = la.norm(M * O1, 'fro')
    X = np.zeros_like(M)
    errors = []
    ranks = []
    while mu >= mu_final:
        #print mu, mu_final
        mu *= eta
        X, cur_errors, cur_ranks = fp_loop(X, M, O1, mu,M_true, I_0)        
        errors += cur_errors
        ranks += cur_ranks
        
    return X, errors, ranks
#-----------------------ALM-----------------------------


def norm12(A):
    s = 0
    for j in xrange(A.shape[1]):
        s += la.norm(A[:,j])
    return s

# thresholding operators
def entry_threshold(A,epsilon):
    return A - np.sign(A)*np.minimum(np.absolute(A),epsilon)

def column_threshold(A,epsilon):
    B = np.zeros(A.shape)
    for j in xrange(A.shape[1]):
        aj = la.norm(A[:,j])
        if aj > epsilon:
            B[:,j] = A[:,j] - epsilon/aj*A[:,j]
    return B

# ALM algorithm for robust matrix completion with corrupted columns
def ALM_RMC(M,Omega,lamda,u0,alpha, M_true, I_0,max_iter = 1e4):
    # M: corrupted matrix(not 0 at other places, in order to compute error)
    # Omega: matrix same size as M, binary entries, =1 at sampled entries
    M_data = M*Omega
    # initialize
    Y = np.zeros(M.shape)
    L = np.zeros(M.shape)
    C = np.zeros(M.shape)
    E = np.zeros(M.shape)
    u = u0 
    Omega0 = (Omega ==0)
    M_F = la.norm(M_data ,'fro') # Frobenius norm of M_data
    print M_F
    k = 0
    #---------- record errors--------
    errors = []
    nuclear_norms = []
    if len(I_0):
        M0 = M_true[:,0:I_0[0]]
    else:
        M0 = M_true
    #--------------------------------
    while la.norm(M_data - E - L - C,'fro')/M_F > 1e-8 and k < max_iter:    
        U,s,V = la.svd(M_data - E - C +1.0/u*Y, full_matrices=False) # s: diagonal vector        
        s_u = entry_threshold(s, 1.0/u)
        
        L = np.dot(U,np.dot(np.diag(s_u),V))
        C = column_threshold(M_data  - E - L + 1.0/u*lamda*Y, 1.0*lamda/u) # a little different from paper(1.0/u*Y)
        E = (M_data - L - C + 1.0/u*Y)*Omega0
#         E[Omega[:,0], Omega[:,1]] = 0
        Y += u*(M_data - E - L - C)
        u = alpha*u
        k += 1
        
        #---------compute errors---------
        #(assume values of M on 0:I_0[0] columns are known)
        if len(I_0):
            L0 = L[:,0:I_0[0]]
        else:
            L0 = L
        err = la.norm(L0 - M0 ,'fro')
        
        errors.append(err/np.size(L0))
        U0,s0,V0 = la.svd(L0)
        nuclear_norms.append(np.sum(s0))
        #--------------------------------
    return L,C,errors,nuclear_norms

def recover_corrupted_columns(C,epsilon = 1e-4):
    # (think of 1e-4 as 0)
    I = []
    C_threshold = entry_threshold(C,epsilon)
    for j in xrange(C_threshold.shape[1]):
        if np.count_nonzero(C_threshold[:,j]) > 0:
            I.append(j)
    return I




