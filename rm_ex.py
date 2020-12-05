import numpy as np
import matplotlib.pyplot as plt


r_Ms = np.array([0.9, 0.8, 0.7])
r_Is = np.array([0.2, 0.5, 0.7])

df = 0.373



f_c = 0.2
f_M = np.linspace(0,1)
DfM = f_M - f_c

plt.figure(figsize=(4, 3))
colors = ['C0', 'C1', 'C3']
for i in range(len(r_Ms)):
    b = DfM*(r_Ms[i] + r_Is[i] - 2)
    plt.plot(DfM, b, label=r'$(r_M,r_I)=' + f'({r_Ms[i]}, {r_Is[i]})' +r'$',
            color=colors[i])
plt.legend(loc='best')
plt.xlabel(r'$f_M - f_{M,c}$')
plt.ylabel(r'Observed differential bias $\hat f_M-\hat f_{M,c}$')

plt.savefig('bias_example.pdf', bbox_inches='tight', dpi=100)



plt.figure()

r_Ms = np.array([0.8, 0.5, 0.7])
r_Is = np.array([0.5, 0.8, 0.7])

f_Ms = np.linspace(0,1)
f_c = 0.2
for i in range(len(r_Ms)):
    fMHat = r_Ms[i]*f_Ms + (1-r_Is[i])*(1-f_Ms)
    fMcHat = r_Ms[i]*f_c + (1-r_Is[i])*(1-f_c)
    plt.plot(f_Ms, fMHat/fMcHat)
plt.plot(f_Ms, f_Ms/f_c, color='k')
plt.show()
