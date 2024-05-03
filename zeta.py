# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def zeta(s, max_iter=500):
    # Calcul de la fonction zêta de Riemann pour un nombre complexe s
    result = 0.0
    for n in range(1, max_iter + 1):
        result += 1.0 / (n ** s)
    return result

# Définir une nouvelle colormap personnalisée avec une transformation non linéaire
def custom_cmap():
    # Création de la liste de couleurs
    colors = [
        (1.0, 1.0, 0.0),   # Vert
        (1.0, 1.0, 1.0),  # Blanc
        (0.0, 0.0, 1.0),  # Bleu
        (0.0, 0.0, 0.0),  # Noir
        (1.0, 1.0, 0.0),  # Jaune
        (1.0, 0.5, 0.0),  # Orange
        (0.0, 1.0, 0.0),  # Rouge
       
    ]

    # Positions des couleurs le long de la colormap (entre 0 et 1)
    positions = [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 1.0]

    # Création de la colormap à partir de la liste de couleurs et positions
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))

    # Fonction de transformation non linéaire pour la colormap
    def cmap_transform(x):
        # Transformation non linéaire pour accentuer les différences près de zéro
        # Utilisation de la fonction logarithmique avec un offset pour éviter les valeurs négatives
        transformed_values = np.log(np.abs(x))
        return np.clip(transformed_values, -10.0, 10.0)  # Clipper les valeurs pour éviter les dépassements

    # Retourner la colormap personnalisée avec la fonction de transformation
    return cmap, cmap_transform

# Définir une grille de points dans le plan complexe
x = np.linspace(-4, 4, 600)
y = np.linspace(-4, 4, 600)
X, Y = np.meshgrid(x, y)
s = X + 1j * Y  # s = x + iy

nbim = 0
mim = 0
# Calculer les valeurs de la fonction zêta de Riemann pour chaque point dans la grille
Z = np.zeros_like(s, dtype=np.complex128)
for i in range(s.shape[0]):
    for j in range(s.shape[1]):
        Z[i, j] = zeta(s[i, j])
        if np.real(Z[i, j]) <= 0.00000000001:
            nbim+=1
            mim += np.real(s[i, j])

print("pour 0.01, nb zero r=", mim/nbim)
print(mim, nbim)

# Créer la nouvelle colormap personnalisée avec la transformation non linéaire
cmap, cmap_transform = custom_cmap()

zero_points = np.where(np.abs(np.real(Z)) == 0.00000000001)

# Appliquer la transformation non linéaire à la partie réelle de Z
Z_real_transformed = cmap_transform(np.real(Z))

# Tracer la partie réelle de la fonction zêta de Riemann dans le plan complexe avec la nouvelle colormap
plt.figure(figsize=(8, 6))
plt.imshow(Z_real_transformed, extent=(np.min(x), np.max(x), np.min(y), np.max(y)),
           cmap=cmap, interpolation='nearest', origin='lower', vmin=-0.5, vmax=1.0)  # Définir vmin et vmax pour la colormap
plt.colorbar(label='Partie réelle (transformée)')
plt.scatter(X[zero_points], Y[zero_points], color='red', s=10, label='Points où Zêta ≈ 0')  # Afficher les points proches de zéro
plt.xlabel('Re(s)')
plt.ylabel('Im(s)')
plt.title('Partie réelle de la fonction zêta de Riemann dans le plan complexe (nuances près de zéro)')
plt.show()
