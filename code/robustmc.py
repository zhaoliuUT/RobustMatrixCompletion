import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pyplot as plt
from functools import partial

from functions_rmc import matrix_psub,matrix_descent,fp_iter,fp_loop,fpc,norm12,entry_threshold,column_threshold,ALM_RMC,recover_corrupted_columns
 

#---------------------------------------------------------------

def random_corrupt(M,gamma,corrupt_value,gamma2 = 0.1):
    #gamma: proportion of corrupted columns. gamma2: proportion of corrupted entries in each column
    M_0 = np.copy(M)
    n_0 = int(gamma*M.shape[1])
    if n_0 == 0:
        I_0 = []
    else:
        I_0 = np.arange(M.shape[1] - n_0, M.shape[1]) # the last several columns are corrupted
        for j in I_0:
            row = rn.randint(0,M.shape[0],int(gamma2*M.shape[0]))
            for i in row:
                M_0[i,j] = corrupt_value
    return M_0, I_0

#------------ main function: only corruption------------
def main_nonoise(T=5000):
    # no noise case
    M = np.genfromtxt('./Data/M.csv', delimiter=',').astype(np.float)
    O = np.genfromtxt('./Data/O.csv', delimiter=',').astype(np.int)  

    GAMMA_LIST = np.linspace(0,0.3,4)
    print GAMMA_LIST
    
    for gamma in GAMMA_LIST:
        # corrupt columns (proportion = gamma)
        print 'gamma = %.1f'%gamma
        M_0, I_0 = random_corrupt(M,gamma,1000)
        print I_0
        if len(I_0):
            U,s,V = la.svd(M[:,0:I_0[0]],full_matrices=False)
        else:
            U,s,V = la.svd(M,full_matrices=False)
        print 'rank of L(noncurrupted part of M) = ', np.sum(s> 1e-3)
        true_nlnorm = int(np.sum(s))*np.ones(T)


        X_m1, error_m1, nlnorms_m1 = matrix_descent(partial(matrix_psub,power=-1.0), M_0, O, M, I_0,T)
        X_m2, error_m2, nlnorms_m2 = matrix_descent(partial(matrix_psub,power=-0.5), M_0, O, M, I_0,T)

        lamda = np.sqrt(M_0.shape[0]**(1.0/4) / M_0.shape[1])*0.5
        alpha = 1.1
        u0 = 1.0/norm12(M_0*O)
        L_m3,C_m3,error_m3,nlnorms_m3 = ALM_RMC(M_0,O,lamda,u0,alpha,M, I_0,T)
        I_m3 = recover_corrupted_columns(C_m3)

        X_m4, error_m4, nlnorms_m4 = fpc(M_0, O,M, I_0)


        plt.clf()
        plt.semilogy(error_m1, label=r'PGD $1/k$')
        plt.semilogy(error_m2, label=r'PGD $1/\sqrt{k}$')
        plt.semilogy(error_m3, label=r'ALM')
        plt.semilogy(error_m4, label=r'FPC')
        plt.title(r'Error on the noncorrupted part: $\gamma$ = %.1f'%gamma)
        plt.legend(loc = 'upper right')
        plt.savefig('error_ns_%d.png'%int(round(gamma*10)))
        plt.show()

        plt.clf()
        plt.plot(nlnorms_m1, label=r'PGD $1/k$')
        plt.plot(nlnorms_m2, label=r'PGD $1/\sqrt{k}$')
        plt.plot(nlnorms_m3, label=r'ALM')
        plt.plot(nlnorms_m4, label=r'FPC')

        plt.plot(true_nlnorm, label=r'$\|L\|_\ast$')

        plt.title(r'Nuclear norm on the noncorrupted part: $\gamma$ = %.1f'%gamma)
        plt.legend(loc = 'lower right')
        plt.savefig('nlnorm_ns_%d.png'%int(round(gamma*10)))

        plt.show()


        plt.clf()
        fig, axarr = plt.subplots(2, sharex=True)
        for xc in I_0:
            axarr[0].axvline(x=xc)
        for xc in I_m3:
            axarr[1].axvline(x=xc)
        axarr[0].set_title('true corrupted locations')
        axarr[1].set_title('recovered corrupted locations')
        axarr[0].set_xlim((0,100));axarr[0].set_yticks([0,1])
        axarr[1].set_xlim((0,100));axarr[1].set_yticks([0,1])
        plt.savefig('column_ns_%d.png'%int(gamma*10))
        plt.show()
        
#-------------main functions: noise+corruption--------------------

def main_noise(T=5000):
    M = np.genfromtxt('./Data/M.csv', delimiter=',').astype(np.float)
    O = np.genfromtxt('./Data/O.csv', delimiter=',').astype(np.int)  


    # add small gaussian noise to M (on observed entries)
    np.random.seed(10)
    G = np.random.normal(loc=0.0, scale=1.0, size=M.shape)
    M_noise = M + G*O

    GAMMA_LIST = np.linspace(0,0.3,4)
    print GAMMA_LIST
    
    for gamma in GAMMA_LIST:
        # corrupt columns (proportion = gamma)
        print 'gamma = %.1f'%gamma
        M_0, I_0 = random_corrupt(M_noise,gamma,1000)
        print I_0
        if len(I_0):
            U,s,V = la.svd(M[:,0:I_0[0]],full_matrices=False)
        else:
            U,s,V = la.svd(M,full_matrices=False)
        print 'rank of L(noncurrupted part of M) = ', np.sum(s> 1e-3)
        true_nlnorm = int(np.sum(s))*np.ones(T)


        X_m1, error_m1, nlnorms_m1 = matrix_descent(partial(matrix_psub,power=-1.0), M_0, O, M, I_0,T)
        X_m2, error_m2, nlnorms_m2 = matrix_descent(partial(matrix_psub,power=-0.5), M_0, O, M, I_0,T)

        lamda = np.sqrt(M_0.shape[0]**(1.0/4) / M_0.shape[1])*0.5
        alpha = 1.1
        u0 = 1.0/norm12(M_0*O)
        L_m3,C_m3,error_m3,nlnorms_m3 = ALM_RMC(M_0,O,lamda,u0,alpha,M, I_0,T)
        I_m3 = recover_corrupted_columns(C_m3)

        X_m4, error_m4, nlnorms_m4 = fpc(M_0, O,M, I_0)


        plt.clf()
        plt.semilogy(error_m1, label=r'PGD $1/k$')
        plt.semilogy(error_m2, label=r'PGD $1/\sqrt{k}$')
        plt.semilogy(error_m3, label=r'ALM')
        plt.semilogy(error_m4, label=r'FPC')
        plt.title(r'Error on the noncorrupted part: $\gamma$ = %.1f'%gamma)
        plt.legend(loc = 'upper right')
        plt.savefig('error_ns_%d.png'%int(round(gamma*10)))
        plt.show()

        plt.clf()
        plt.plot(nlnorms_m1, label=r'PGD $1/k$')
        plt.plot(nlnorms_m2, label=r'PGD $1/\sqrt{k}$')
        plt.plot(nlnorms_m3, label=r'ALM')
        plt.plot(nlnorms_m4, label=r'FPC')

        plt.plot(true_nlnorm, label=r'$\|L\|_\ast$')

        plt.title(r'Nuclear norm on the noncorrupted part: $\gamma$ = %.1f'%gamma)
        plt.legend(loc = 'lower right')
        plt.savefig('nlnorm_ns_%d.png'%int(round(gamma*10)))

        plt.show()


        plt.clf()
        fig, axarr = plt.subplots(2, sharex=True)
        for xc in I_0:
            axarr[0].axvline(x=xc)
        for xc in I_m3:
            axarr[1].axvline(x=xc)
        axarr[0].set_title('true corrupted locations')
        axarr[1].set_title('recovered corrupted locations')
        axarr[0].set_xlim((0,100));axarr[0].set_yticks([0,1])
        axarr[1].set_xlim((0,100));axarr[1].set_yticks([0,1])
        plt.savefig('column_ns_%d.png'%int(gamma*10))
        plt.show()


if __name__ == "__main__":
    main_nonoise(5000)
    main_noise(5000)
