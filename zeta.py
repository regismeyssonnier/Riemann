# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpmath

def sieve_of_eratosthenes(n):
    # Initialiser un tableau de booléens pour marquer les nombres premiers
    is_prime = [True] * (n + 1)
    p = 2
    while (p * p <= n):
        if (is_prime[p] == True):
            # Marquer comme non premier tous les multiples de p commençant par p*p
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    
    # Collecter tous les nombres premiers dans une liste
    prime_numbers = []
    for p in range(2, n + 1):
        if is_prime[p]:
            prime_numbers.append(p)
    
    return prime_numbers


prime = sieve_of_eratosthenes(1000000)

def zeta(s, max_iter=1000):
    # Calcul de la fonction zêta de Riemann pour un nombre complexe s
    result = 0.0
    for n in range(1, max_iter + 1):
        result = 1.0 / (n ** s)
    return result

def zeta3(s, max_iter=1000):
    # Calcul de la fonction zêta de Riemann pour un nombre complexe s
    result = 0.0
    for n in range(1, max_iter + 1):
        result = 1.0 / np.exp(-s * np.log(n))
    return result

def zeta2(s, max_iter=100):
    # Calcul de la fonction zêta de Riemann pour un nombre complexe s
    result = 0.0
    for n in range(0, max_iter):
        if n == 0:
            result = 1.0 / (1.0 - 1.0/(prime[n] ** s))
        else:
            result *= 1.0 / (1.0 - 1.0/(prime[n] ** s))
    return result

def zetal(s, max_iter=100):
    # Calcul de la fonction zêta de Riemann pour un nombre complexe s
    result = 0.0
    for n in range(0, max_iter):
      result +=np.log(1.0 - 1.0/(prime[n] ** s))
    return -result

def zetam(s):
    return mpmath.zeta(s)

# Définir une nouvelle colormap personnalisée avec une transformation non linéaire
def custom_cmap():
    # Création de la liste de couleurs
    colors = [
        (1.0, 0.0, 0.0),   # Vert
        (1.0, 1.0, 1.0),  # Blanc
        (0.0, 0.0, 1.0),  # Bleu
        (0.0, 0.0, 0.0),  # Noir
        (1.0, 1.0, 0.0),  # Jaune
        (1.0, 0.5, 0.0),  # Orange
        (0.0, 1.0, 0.0),  # Rouge
       
    ]

    # Positions des couleurs le long de la colormap (entre 0 et 1)
    positions = [0.0, 0.2, 0.4, 0.5, 0.8, 0.85, 1.0]

    # Création de la colormap à partir de la liste de couleurs et positions
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))

    # Fonction de transformation non linéaire pour la colormap
    def cmap_transform(x):
        # Transformation non linéaire pour accentuer les différences près de zéro
        # Utilisation de la fonction logarithmique avec un offset pour éviter les valeurs négatives
        return x #np.log(np.abs(x)+1.0)
        #return np.clip(x, -1.0, 1.0)  # Clipper les valeurs pour éviter les dépassements

    # Retourner la colormap personnalisée avec la fonction de transformation
    return cmap, cmap_transform

# Définir une grille de points dans le plan complexe
x = np.linspace(-40, 40, 500)
y = np.linspace(-40, 40, 500)
X, Y = np.meshgrid(x, y)
s = X + 1j * Y  # s = x + iy

print(zetam(-2), zeta(-4), -1.0/12.0)

print(len(prime))
print(prime[0:10])

nbim = 0
mim = 0
# Calculer les valeurs de la fonction zêta de Riemann pour chaque point dans la grille
Z = np.zeros_like(s, dtype=np.complex128)
ZR = np.zeros((s.shape[0], s.shape[1]))
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        z = zetam(s[i, j])
        #print(z)
        #mz =  np.sqrt(np.real(z)**2 + np.imag(z)**2)
        Z[i, j] = z
        #print(mz, Z[i, j])
        if Z[i, j] < 0.001:
            nbim+=1
            mim += np.real(s[i, j])

if nbim > 0:
    print("pour 0.01, nb zero r=", mim/nbim)
    print(mim, nbim)

# Créer la nouvelle colormap personnalisée avec la transformation non linéaire
cmap, cmap_transform = custom_cmap()

zero_points = np.where(np.abs(np.real(Z)) < 0.001)
#zero_points = np.where(np.abs(ZR) < 0.001)


# Appliquer la transformation non linéaire à la partie réelle de Z
Z_real_transformed = cmap_transform(np.real(Z))
#Z_real_transformed = cmap_transform(ZR)

# Tracer la partie réelle de la fonction zêta de Riemann dans le plan complexe avec la nouvelle colormap
plt.figure(figsize=(8, 6))
plt.imshow(Z_real_transformed, extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
           cmap=cmap, interpolation='nearest', origin='lower', vmin=-1.0, vmax=1.0)  # Définir vmin et vmax pour la colormap
plt.colorbar(label='Partie réelle (transformée)')
plt.scatter(X[zero_points], Y[zero_points], color='black', s=10, label='Points où Zêta ≈ 0')  # Afficher les points proches de zéro
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
plt.title('Partie réelle de la fonction zêta de Riemann dans le plan complexe (nuances près de zéro)')
plt.show()
