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


    # Expected number of correctly identified mergers
    N_M1 = r_M * N_M
    # Expected number of isolated galaxies identified as merger
    N_M2 = (1-r_I)*(N-N_M)


    f_M = N_M/N
    N_input, N_Mhats, r_Ms, r_Is = make_sample(N, f_M, n_survey, N_M=0, width_I=0.00,
            width_M=0.00, I=r_I, M=r_M)
    bins = np.arange(N+1)-0.5
    plt.hist(N_Mhats, bins=bins, density=True)
    plt.axvline(N_M, label='Number of input mergers', color='b')
    plt.axvline(N_M1 + N_M2, label='Expected number of measured mergers',
            color='r')
    plt.legend(loc='best')
    plt.xlabel(r'$\hat N_M$')
    plt.ylabel(r'$P(\hat N_M\mid N_M, N_I, r_I, r_M$')
    return


def test2(N_M=0, N=10, r_I=0.7, r_M=0.8, n_survey=25):
    '''
    Given a measurement N_Mhat, what is the probability that N_M is recovered?
    '''
    f_M = N_M/N
    N_input, N_Mhats, r_Ms, r_Is = make_sample(N, f_M, n_survey, N_M=N_M, width_I=0.00,
            width_M=0.00, I=r_I, M=r_M)



    Lis = []
    for N_Mhat in N_Mhats:
        N_Ms = np.arange(N+1)
        Lis.append(L(N_Ms, r_I, r_M, N_Mhat, N))
        plt.plot(N_Ms, Lis[-1], color='k', alpha=0.5, zorder=2)

    Lis = np.array(Lis)
    logL_tot = sum(np.log(Lis))
    L_tot = np.exp(logL_tot - logL_tot.max())
    plt.plot(N_Ms, L_tot, lw=5)
    plt.axvline(N_M, linestyle='--', label='True Input')
    plt.xlabel(r'$N_M$')
    plt.ylabel(r'$P(N_M\mid \{\hat N_M\},\{r_I\},\{r_J\}, N_\mathrm{tot})$')
    plt.legend(loc='best')

    return


def test3(f_M=0.3):
    '''
    Attempting to show how off a merger fraction could be as a function of
    accuracy.
    '''
    r_M = np.linspace(0,1)
    r_I = np.linspace(0,1)
    r_M, r_I = np.meshgrid(r_M, r_I)


    f_MHat = r_M*f_M + (1-r_I)*(1-f_M)
    plt.figure()
    plt.pcolormesh(r_M, r_I, f_MHat, vmin=0, vmax=1)
    plt.colorbar(label=r'$\hat f_M$')
    plt.xlabel(r'$r_M$')
    plt.ylabel(r'$r_I$')
    plt.title(r'$f_M={0}$'.format(f_M))
    return



def test4(f_M=0.3, N=100):
    '''
    Attempting to show how off a merger fraction could be as a function of
    accuracy.
    '''
    r_M = np.linspace(0,1)
    r_I = np.linspace(0,1)
    r_M, r_I = np.meshgrid(r_M, r_I)

    N_M = f_M*N
    N_I = (1-f_M)*N



    N_MHat = r_M*N_M + (1-r_I)*N_I
    plt.figure()
    plt.pcolormesh(r_M, r_I, N_MHat)
    plt.colorbar(label=r'$\hat N_M$')
    plt.xlabel(r'$r_M$')
    plt.ylabel(r'$r_I$')
    plt.title(r'$f_M={0}$'.format(f_M))
    return



def test5():
    '''
    I would like to get the probability that a galaxy is a merger given an
    individual's classification.
    '''
    # Let's say a respondent says that a given galaxy is a merger.
    r_M = np.linspace(0,1)
    r_I = 0.8
    p_Mhat_M = r_M
    p_Mhat_I = 1-r_I

    p_M = p_Mhat_M/(p_Mhat_M + p_Mhat_I)
    plt.plot(r_M, p_M)
    plt.xlabel(r'$r_M$')
    plt.ylabel(r'$p(\mathrm{{merger}}\mid\mathrm{{observed merger}},r_I={0},r_M)$'.format(r_I))
    

    plt.figure()
    r_M = 0.7
    r_I = np.linspace(0,1)
    p_Mhat_M = r_M
    p_Mhat_I = 1-r_I

    p_M = p_Mhat_M/(p_Mhat_M + p_Mhat_I)
    plt.plot(r_I, p_M)
    plt.xlabel(r'$r_I$')
    plt.ylabel(r'$p(\mathrm{{merger}}\mid\mathrm{{observed merger}},r_M={0},r_I)$'.format(r_M))
    plt.show()

    return



def test6():
    '''
    Let us simulate a group of observers with known accuracies, and get the
    probability of a galaxy actually being a merger.
    '''
    N = 1
    N_M = 1
    f_M = 1
    n_survey = 5
    N_input, N_Mhats, r_Ms, r_Is = make_sample(N, f_M, n_survey, N_M=N_M, I=0.8,
            M=0.8)
    p_Mhat_M = r_Ms
    p_Mhat_I = 1-r_Is
    p_M = p_Mhat_M/(p_Mhat_M + p_Mhat_I)
    plt.hist(p_M)


    print(N_input)
    print(N_Mhats)
    print(r_Ms)
    print(r_Is)


    P = 1
    den_M = 1
    den_I = 1
    for i in range(len(N_Mhats)):
        den_M *= r_Ms[i]
        den_I *= 1-r_Is[i]
        P *= r_Ms[i]
    den = den_M + den_I
    print('Probability of being a merger', P/den)
    print('Average answer', N_Mhats.mean())
    plt.show()

    return

if __name__ == '__main__':
    '''
    N = 50
    Nhat = np.arange(N+1)
    r_M = 0.8
    r_I = 0.7
   
    plt.figure()
    test1(N=10, N_M=0, r_M=r_M, r_I=r_I)
    plt.figure()
    test1(N=20, N_M=0, r_M=r_M, r_I=r_I)
    plt.figure()
    test1(N=100, N_M=0, n_survey=2000)
    plt.show()

    test2(N=50, N_M=50)
    plt.show()

    test3()
    test3(f_M=0.1)
    test3(f_M=0.7)
    plt.show()
    '''

    test6()
