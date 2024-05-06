def binomial_coefficient(n, k):
    """ Calculer le coefficient binomial C(n, k) """
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    numerator = 1
    denominator = 1
    
    # Calcul de n * (n-1) * ... * (n-k+1)
    for i in range(1, k + 1):
        numerator *= (n - (k - i))
    
    # Calcul de k!
    for i in range(1, k + 1):
        denominator *= i
    
    # Calcul du coefficient binomial
    return numerator // denominator

def precalculate_bernoulli_numbers(N):
    """ Pré-calculer les nombres de Bernoulli jusqu'au rang N """
    bernoulli = [0] * (N + 1)
    bernoulli[0] = 1  # B_0 = 1
    
    for n in range(1, N + 1):
        sum_term = 0
        for k in range(n):
            sum_term += binomial_coefficient(n + 1, k) * bernoulli[k]
        bernoulli[n] = -sum_term / (n + 1)
    
    return bernoulli

# Définir le rang maximal des nombres de Bernoulli à calculer
max_n = 100

# Pré-calculer les nombres de Bernoulli jusqu'au rang max_n
bernoulli_sequence = precalculate_bernoulli_numbers(max_n)

# Afficher les nombres de Bernoulli calculés
for n in range(max_n + 1):
    print(f"B_{n} = {bernoulli_sequence[n]}")

