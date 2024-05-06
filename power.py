import numpy as np

def complex_power(n, s):
    # Calcul de n^s ou s est un nombre complexe
    return np.exp(s * np.log(n))

# Exemple d'utilisation :
n = 2
s = 1 + 2j  # s est un nombre complexe

result = complex_power(n, s)
print(f"{n}^{s} =", result, n**s)