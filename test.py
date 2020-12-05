import accuracy_analysis as aa
import numpy as np
import matplotlib.pyplot as plt

N = 50
# number of classifier
n = 14

n_exp = 100

f_M_truths = np.arange(0.05, 0.95, 0.1)
fs = np.arange(0.,N+1)/N

Z1mu = []
Z2mu = []
Z1sd = []
Z2sd = []
r_M_base = 0.9
r_I_base = 0.5
#fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
#axs = axes.flatten()
for f_i in range(len(f_M_truths)):
    mu1 = np.zeros(n_exp)
    mu2 = np.zeros(n_exp)
    ll1 = np.zeros(n_exp)
    ul1 = np.zeros(n_exp)
    ll2 = np.zeros(n_exp)
    ul2 = np.zeros(n_exp)

    N_trues = np.zeros(n_exp)
    Z1 = np.zeros(n_exp)
    Z2 = np.zeros(n_exp)

    Ns = N*fs
    for _ in range(n_exp):
        N_true, N_M, r_Ms, r_Is = aa.make_sample(N, f_M_truths[f_i], n, r_M=r_M_base, r_I=r_I_base)
        P = aa.L(Ns, r_Ms[0], r_Is[0], N_M[0], N)*0 + 1
        for i in range(n):
            P *= aa.L(Ns, r_Is[i], r_Ms[i], N_M[i], N)
        #P = np.exp(lnLi - lnLi.max())
        mu = sum(fs*P)/sum(P)
        ll, ul, sd = aa.pdf_std(P, fs)
        Z1[_] = (mu-N_true/N)/sd
        mu1[_] = mu
        ll1[_] = ll
        ul1[_] = ul

        mu = (n*N_M.mean()+1)/(n*N+2)
        mu2[_] = mu
        #ll, ul, sd = beta_std(sum(N_M), n*N)
        ll, ul, sd = aa.beta_std(N_M.mean(), N)
        Z2[_] = (mu-N_true/N)/sd
        ll2[_] = ll
        ul2[_] = ul

        N_trues[_] = N_true
    #if f_i == 0:
    #    bins = np.linspace(-int(3*Z2.mean()), int(3*Z2.mean()), 101)

    #axs[f_i].hist(Z1, bins=bins, alpha=0.5)
    #axs[f_i].hist(Z2, bins=bins, alpha=0.5)
    Z1mu.append(Z1.mean())
    Z2mu.append(Z2.mean())
    Z1sd.append(Z2.std())
    Z2sd.append(Z2.std())

# Make plot of merger fraction on horizontal axis, y axis is mean sigma away, error bars are spread of distribution.
plt.figure()
plt.errorbar(f_M_truths, Z1mu, Z1sd, fmt='o', label='Merger Fraction Likelihood Method')
plt.errorbar(f_M_truths, Z2mu, Z2sd, fmt='o', label='Standard Binomial Method')
plt.legend()
plt.ylabel(r'Distance From True Merger Fraction ($\sigma$)')
plt.xlabel(r'$f_M$')

plt.savefig('test.png', bbox_inches='tight')
plt.savefig('test.pdf', bbox_inches='tight')
plt.savefig('sigma_difference_comparison.pdf', bbox_inches='tight')

plt.show()
