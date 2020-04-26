import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multinomial, beta, dirichlet
from scipy.special import binom, loggamma
from scipy.integrate import cumtrapz
import statsmodels.stats.proportion

# progress bar
from tqdm import tqdm
import pandas as pd

plot_all_lines = False
import warnings
warnings.simplefilter('error', RuntimeWarning)


def beta_std(k, N, alpha=0.32, method='beta'):
    ll, ul = statsmodels.stats.proportion.proportion_confint(k, N, alpha=alpha, method=method)
    return ll, ul, (ul-ll)/2

def pdf_std(pdf, x, alpha=0.32):
    cdf = cumtrapz(pdf,x)/cumtrapz(pdf,x)[-1]
    if len(x[1:][cdf < (0.5-(1-alpha)/2)]) > 0:
        ll = x[1:][cdf < (0.5-(1-alpha)/2)][-1]
    else:
        ll = x[0]
    ul = x[1:][cdf > (0.5+(1-alpha)/2)][0]
    return ll, ul, (ul-ll)/2
    
def make_sample(N, f_M, n, N_M=None, width_I=0.05, width_M=0.05, I=0.6, M=0.8):
    if N_M is None:
        true_gals = np.random.choice([0,1], size=N, p=[1-f_M, f_M])
        true_gals.sort()
    else:
        true_gals = np.array([0]*(N-N_M) + [1]*N_M)
    f_M_sample = true_gals.sum()/N
    N_true = true_gals.sum()
    
    r_Ms = np.random.uniform(low=M-width_M/2, high=M+width_M/2, size=n)
    r_Is = np.random.uniform(low=I-width_I/2, high=I+width_I/2, size=n)

    # Matrix of classifier answers
    m = np.zeros((n, N), dtype='int')
    for i in range(n):
        for j in range(N):
            if true_gals[j] == 0:
                m[i,j] = np.random.choice([0,1], p=[r_Is[i], 1-r_Is[i]])
            elif true_gals[j] == 1:
                m[i,j] = np.random.choice([0,1], p=[1-r_Ms[i], r_Ms[i]])
    N_M = m.sum(axis=1)

    
    return N_true, N_M, r_Ms, r_Is


def logbinom(n, k):
    logbinom = loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)
    #logbinom[n<k] = -np.inf
    #logbinom[k<0] = -np.inf
    return logbinom

def logbinom_lnnk(n,k):
    logbinom = k*np.log(n/k - 0.5) + k -0.5*np.log(2*np.pi*k)
    return logbinom
    
def f_b(k, n, p, thresh=100):
    # k is an integer, n is an array
    '''
    if (p == 0) or (p == 1) or (k < 0):
        return -np.inf
    '''
    logf1 = logbinom(n, k)

    logf2 = k*np.log(p)
    logf3 = (n-k)*np.log(1-p)
    logf_b = logf1+logf2+logf3
    #logf_b = logf1
    return logf_b

'''
def L(Ns, rj, rjp, N_M, Ntot, high_acc=False):
    L = 0
    kmin = Ns.min()+N_M-Ntot
    kmax = Ntot
    print(kmin, kmax)
    if high_acc:
        #N_Mks = np.linspace(kmin,kmax,1001)
        N_Mks = np.linspace(0,Ntot,1001)
    else:
        #Ns = Ns.astype(int)
        #N_Mks = np.arange(int(kmin),int(kmax+1))
        N_Mks = np.arange(0.5,Ntot+1)
    for k in N_Mks:
        L1 = np.zeros(len(Ns))-np.inf
        L2 = np.zeros(len(Ns))-np.inf
        inds = ((Ns >= k) & ((Ntot-Ns) >= (N_M-k)))
        L1[inds] = f_b(k, Ns[inds], rj)
        L2[inds] = f_b(N_M-k, Ntot-Ns[inds], 1-rjp)
        L += np.exp(L1+L2)
    return L/sum(L*np.diff(Ns)[0])
'''

def P(N_Mhat, N_M, N_I, r_I, r_M):
    P = 0
    for N_Mhat1 in np.arange(0, N_M+1):
        N_Mhat2 = N_Mhat - N_Mhat1
        P1 = binom(N_M, N_Mhat1)*r_M**N_Mhat1*(1-r_M)**(N_M-N_Mhat1)
        P2 = binom(N_I, N_I-N_Mhat2)*r_I**(N_I-N_Mhat2)*(1-r_I)**N_Mhat2
        P += P1*P2
    return P


def L(N_M, r_I, r_M, N_Mhat, N_tot):
    N_I = N_tot - N_M
    L = np.zeros(len(N_M))
    for i in range(len(L)):
        L[i] = P(N_Mhat, N_M[i], N_I[i], r_I, r_M)
    #return L/sum(L*np.diff(N_M)[0])
    return L/max(L)


def test1(N_M=0, N=10, r_I=0.7, r_M=0.8,
        n_survey=1000):
    '''
    Simulate many datasets and see that the expected number of observed mergers
    is correctly predicted by the probability distribution.
    '''
    N_Mhats = np.linspace(0, N)
    P_Nhats = np.zeros_like(N_Mhats)
    for i in range(len(N_Mhats)):
        P_Nhats[i] = P(N_Mhats[i], N_M, N-N_M, r_I, r_M)
    plt.plot(N_Mhats, P_Nhats)
    plt.xlabel(r'$\hat N_M$')


    f_M = N_M/N
    N_input, N_Mhats, r_Ms, r_Is = make_sample(N, f_M, n_survey, N_M=0, width_I=0.00,
            width_M=0.00, I=r_I, M=r_M)
    bins = np.arange(N+1)-0.5
    plt.hist(N_Mhats, bins=bins, density=True)
    return


def test2(N_M=0, N=10, r_I=0.7, r_M=0.8):
    '''
    Given a measurement N_Mhat, what is the probability that N_M is recovered?
    '''
    f_M = N_M/N
    n_survey = 25
    N_input, N_Mhats, r_Ms, r_Is = make_sample(N, f_M, n_survey, N_M=0, width_I=0.00,
            width_M=0.00, I=r_I, M=r_M)



    for N_Mhat in N_Mhats:
        N_Ms = np.arange(N+1)
        plt.plot(N_Ms, L(N_Ms, r_I, r_M, N_Mhat, N), color='k', alpha=0.5)


    return

if __name__ == '__main__':
    N = 50
    Nhat = np.arange(N+1)
    r_M = 0.8
    r_I = 0.7
   
    '''
    plt.figure()
    test1(N=10, N_M=0, r_M=r_M, r_I=r_I)
    plt.figure()
    test1(N=20, N_M=0, r_M=r_M, r_I=r_I)
    plt.figure()
    test1(N=100, N_M=0, n_survey=2000)
    plt.show()
    '''

    test2(N=100)
    plt.show()



