# graphical model for likelihood


# Example for weak lensing.
import daft


pgm = daft.PGM()
pgm.add_node("Omega", r"$\Omega$", -1, 4)
pgm.add_node("gamma", r'$\gamma$', 0, 4)
pgm.add_node("obs", r"$\epsilon^{\mathrm{obs}}_n$", 1, 4, observed=True)
pgm.add_node("alpha", r"$\alpha$", 3, 4)
pgm.add_node("true", r"$\epsilon^{\mathrm{true}}_n$", 2, 4)
pgm.add_node("sigma", r"$\sigma_n$", 1, 3)
pgm.add_node("Sigma", r"$\Sigma$", 0, 3)
pgm.add_node("x", r"$x_n$", 2, 3, observed=True)
pgm.add_plate([0.5, 2.25, 2, 2.25], label=r"galaxies $n$")
pgm.add_edge("Omega", "gamma")
pgm.add_edge("gamma", "obs")
pgm.add_edge("alpha", "true")
pgm.add_edge("true", "obs")
pgm.add_edge("x", "obs")
pgm.add_edge("Sigma", "sigma")
pgm.add_edge("sigma", "obs")

pgm.render()
pgm.savefig("weaklensing.pdf")


# Make a model for a single galaxy, or a single classifier, or the whole sample.

# N, single observer. To start, let's ignore the r_I, r_M things.

pgm = daft.PGM()
pgm.add_node("NM", r"$N_M$", -1, 4)
pgm.add_node('NMhat1', r'$\hat N_{M,1}$', 0, 4.5)
pgm.add_node('NMhat2', r'$\hat N_{I,2}$', 0, 3.5)
pgm.add_node("NMhat", r'$\hat N_M$', 1, 4, observed=True)
pgm.add_node("NI", r"$N_I$", -1, 2)
pgm.add_node('NIhat1', r'$\hat N_{I,1}$', 0, 2.5)
pgm.add_node('NIhat2', r'$\hat N_{M,2}$', 0, 1.5)
pgm.add_node("NIhat", r'$\hat N_I$', 1, 2, observed=True)


pgm.add_edge('NM', 'NMhat1')
pgm.add_edge('NM', 'NMhat2')
pgm.add_edge('NI', 'NIhat1')
pgm.add_edge('NI', 'NIhat2')

pgm.add_edge('NMhat1', 'NMhat')
pgm.add_edge('NIhat2', 'NMhat')
pgm.add_edge('NIhat1', 'NIhat')
pgm.add_edge('NMhat2', 'NIhat')

pgm.add_plate([-0.5, 0.75, 2.0, 4.25], label='observers $i$')

pgm.render()
pgm.savefig("model1.pdf")



pgm = daft.PGM()

pgm.add_node('fM', r'$f_M$', -2, 3)

pgm.add_node("NM", r"$N_M$", -1, 4)
pgm.add_node('rM', r'$r_M$', 0.5, 4)
pgm.add_node('NMhat1', r'$\hat N_{M,1}$', 1, 4.5)
pgm.add_node('NMhat2', r'$\hat N_{I,2}$', 1, 3.5)
pgm.add_node("NMhat", r'$\hat N_M$', 2, 4, observed=True)

pgm.add_node("NI", r"$N_I$", -1, 2)
pgm.add_node('rI', r'$r_I$', 0.5, 2)
pgm.add_node('NIhat1', r'$\hat N_{I,1}$', 1, 2.5)
pgm.add_node('NIhat2', r'$\hat N_{M,2}$', 1, 1.5)
pgm.add_node("NIhat", r'$\hat N_I$', 2, 2, observed=True)

pgm.add_node("fMhat", r'$\hat f_M$', 2.5, 3)

pgm.add_edge('fM', 'NM')
pgm.add_edge('fM', 'NI')

pgm.add_edge('NM', 'NMhat1')
pgm.add_edge('NM', 'NMhat2')
pgm.add_edge('NI', 'NIhat1')
pgm.add_edge('NI', 'NIhat2')
pgm.add_edge('rM', 'NMhat1')
pgm.add_edge('rM', 'NMhat2')
pgm.add_edge('rI', 'NIhat1')
pgm.add_edge('rI', 'NIhat2')

pgm.add_edge('NMhat1', 'NMhat')
pgm.add_edge('NIhat2', 'NMhat')
pgm.add_edge('NIhat1', 'NIhat')
pgm.add_edge('NMhat2', 'NIhat')


pgm.add_edge('NMhat', 'fMhat')
pgm.add_edge('NIhat', 'fMhat')

pgm.add_plate([-0.5, 0.75, 3.5, 4.25], label='observer $i$')
pgm.add_plate([-1.5, 0.5, 4.75, 4.75], label='sample $n$')
pgm.render()
pgm.savefig("model2.pdf")



pgm = daft.PGM()

pgm.add_node('fM', r'$f_M$', -2, 3)

pgm.add_node("NM", r"$N_M$", -1, 4)
pgm.add_node('rM', r'$r_M$', 0.5, 4)
pgm.add_node('NMhat1', r'$\hat N_{M,1}$', 1, 4.5)
pgm.add_node('NMhat2', r'$\hat N_{I,2}$', 1, 3.5)
pgm.add_node("NMhat", r'$\hat N_M$', 2, 4, observed=True)

pgm.add_node("NI", r"$N_I$", -1, 2)
pgm.add_node('rI', r'$r_I$', 0.5, 2)
pgm.add_node('NIhat1', r'$\hat N_{I,1}$', 1, 2.5)
pgm.add_node('NIhat2', r'$\hat N_{M,2}$', 1, 1.5)
pgm.add_node("NIhat", r'$\hat N_I$', 2, 2, observed=True)



pgm.add_node("NMfake", r'$N_{M, \mathrm{sim}}$', -1, 3.33)
pgm.add_node("NIfake", r'$N_{I, \mathrm{sim}}$', -1, 2.66)

pgm.add_node("rMhat", r'$\hat r_M$', 0.25, 3.33, observed=True)
pgm.add_node("rIhat", r'$\hat r_I$', 0.25, 2.66, observed=True)

pgm.add_node("fMhat", r'$\hat f_M$', 2.85, 3)


pgm.add_edge('NMfake', 'rMhat')
pgm.add_edge('NIfake', 'rIhat')

pgm.add_edge('fM', 'NM')
pgm.add_edge('fM', 'NI')

pgm.add_edge('NM', 'NMhat1')
pgm.add_edge('NM', 'NMhat2')
pgm.add_edge('NI', 'NIhat1')
pgm.add_edge('NI', 'NIhat2')
pgm.add_edge('rM', 'NMhat1')
pgm.add_edge('rM', 'NMhat2')
pgm.add_edge('rI', 'NIhat1')
pgm.add_edge('rI', 'NIhat2')

pgm.add_edge('NMhat1', 'NMhat')
pgm.add_edge('NIhat2', 'NMhat')
pgm.add_edge('NIhat1', 'NIhat')
pgm.add_edge('NMhat2', 'NIhat')

pgm.add_edge('NMhat', 'fMhat')
pgm.add_edge('NIhat', 'fMhat')
pgm.add_edge('rMhat', 'fMhat')
pgm.add_edge('rIhat', 'fMhat')
pgm.add_edge('rM', 'fMhat')
pgm.add_edge('rI', 'fMhat')
pgm.add_edge('fMhat', 'rM')
pgm.add_edge('fMhat', 'rI')



pgm.add_plate([-0.5, 0.75, 3, 4.25], label='observer $i$')
pgm.add_plate([-1.5, 0.5, 4.75, 4.75], label='sample $n$')
pgm.render()
pgm.savefig("model3.pdf")



