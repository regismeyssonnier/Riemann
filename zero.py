import mpmath

def find_nontrivial_zeros(max_iter=100, tol=1e-15):
    """ Trouver les zéros non triviaux de la fonction zêta de Riemann dans la bande critique """
    mpmath.mp.dps = 25  # Précision décimale arbitrairement élevée pour une bonne précision
    
    zeros = []
    zmill = []
    n = 1
    while len(zeros) < max_iter:
        # Calculer le n-ième zéro non trivial sur la ligne critique Re(s) = 0.5
        zero = mpmath.zetazero(n)
        
        # Vérifier si le zéro trouvé est sur la ligne critique
        if mpmath.re(zero) == 0.5:
            zeros.append(zero)
        else:
            zmill.append(zero)
        
        n += 1
    
    return zeros, zmill

# Trouver les premiers 10 zéros non triviaux de la fonction zêta de Riemann
zeros, zmill = find_nontrivial_zeros(max_iter=100)

# Afficher les zéros non triviaux trouvés
for i, zero in enumerate(zeros):
    print(f"Zéro non trivial {i+1}: {zero}")

for i, zero in enumerate(zmill):
    print(f"Zéro non trivial 1M {i+1}: {zero}")