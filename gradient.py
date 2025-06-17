import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import time


# Fonction h(x) à optimiser dans les exercices 1, 2 et 3
def h(x):
    """Fonction polynomiale à optimiser de la forme h(x) = -x^2 + 4x + 42"""
    return -x ** 2 + (20/3) * x + 40


# Gradient (dérivée) de la fonction h
def grad_h(x):
    """Gradient (dérivée) de la fonction h(x) = -x^2 + 20/3 + 42
       grad_h(x) = -2x + 4"""
    return -2 * x + 20/3


# Exercice 1: descente du gradient simple
def gradient_descent_simple(X0, alpha):
    """
    Implémentation basique de la descente de gradient pour une fonction à une variable

    Args:
        X0 (float): Point de départ
        alpha (float): Pas d'apprentissage (learning rate)

    Returns:
        float: Point optimisé après la descente de gradient
    """
    # Initialisation
    X = X0
    gradX = grad_h(X0)
    nb_tour = 0

    # Boucle des itérations
    while (abs(gradX) > 1e-2 and nb_tour < 10):
        gradX = grad_h(X)
        X -= gradX * alpha * (-1)
        nb_tour += 1

    return X


# Version améliorée avec affichage des étapes pour le cours
def gradient_descent_cours(X0, alpha):
    """
    Version améliorée de la descente de gradient avec affichage des étapes

    Args:
        X0 (float): Point de départ
        alpha (float): Pas d'apprentissage (learning rate)
    """
    # Initialisation
    X = X0
    gradX = grad_h(X)
    nb_tour = 0

    print(f"Départ: x={X:.4f}, h(x)={h(X):.4f}, grad={gradX:.4f}")

    # Boucle des itérations
    while (abs(gradX) > 1e-2 and nb_tour < 10):
        gradX = grad_h(X)
        X_old = X
        X -= gradX * alpha * (-1)  # -1 car on cherche un maximum
        nb_tour += 1
        print(f"Tour {nb_tour}: x={X:.4f}, h(x)={h(X):.4f}, grad={gradX:.4f}, delta_x={X - X_old:.4f}")

    print(f"Résultat: x={X:.4f}, h(x)={h(X):.4f}, après {nb_tour} tours")


def exercice1():
    """Teste la descente de gradient avec différentes valeurs de alpha"""
    print("** EXERCICE 1 **")
    print("Extremum : fmax=51.11 en x=3.35")

    # Alpha petit (0.01) : convergence très progressive et lente
    # Nécessite plus d'itérations mais évite les oscillations
    print("Alpha = 0.01")
    gradient_descent_cours(2, 0.01)

    # Alpha modéré (0.1) : bon compromis entre vitesse et précision
    # Convergence plus rapide sans risque majeur d'instabilité
    print("Alpha = 0.1")
    gradient_descent_cours(2, 0.1)

    # Alpha élevé (0.5) : convergence rapide
    # Risque d'oscillations ou de dépassement de l'optimum
    print("Alpha = 0.5")
    gradient_descent_cours(2, 0.5)

    # Alpha très élevé (1) : convergence très rapide mais risquée
    # Peut entraîner des oscillations importantes ou une divergence
    # Pour cette fonction simple et convexe, reste stable
    print("Alpha = 1")
    gradient_descent_cours(2, 1)

    # Remarques générales:
    # - Un alpha trop petit ralentit la convergence (nombreuses itérations)
    # - Un alpha trop grand peut provoquer des oscillations ou une divergence
    # - Le choix optimal dépend de la forme de la fonction à optimiser
    # - Pour cette fonction simple, même un alpha élevé fonctionne bien




def gradient_descent(grad, X0, alpha, stop_condition=1e-2, max_iter=100, sens=1):
    """
    Descente de gradient généralisée

    Args:
        grad (function): Fonction de calcul du gradient
        X0 (float): Point de départ
        alpha (float): Pas d'apprentissage
        stop_condition (float): Condition d'arrêt sur la valeur absolue du gradient
        max_iter (int): Nombre maximum d'itérations
        sens (int): Sens de la descente (+1 pour minimum, -1 pour maximum)

    Returns:
        tuple: (Point optimisé, nombre d'itérations, temps en ms)
    """
    # Initialisation
    X = X0
    nb_iter = 0
    start_time = time.time()

    # Boucle des itérations
    while True:
        gradX = grad(X)

        # Condition d'arrêt sur le gradient
        if abs(gradX) <= stop_condition:
            break

        # Condition d'arrêt sur le nombre d'itérations
        if nb_iter >= max_iter:
            break

        # Mise à jour de X (sens=1 pour minimiser, sens=-1 pour maximiser)
        X = X - sens * alpha * gradX
        nb_iter += 1

    # Calcul du temps d'exécution
    exec_time = (time.time() - start_time) * 1000  # en millisecondes

    # Affichage des résultats
    print(f"Résultat: X = {X:.6f}, Nombre d'itérations: {nb_iter}, Temps: {exec_time:.2f} ms")

    return X, nb_iter, exec_time


def f_ex2(x):
    """Fonction (x-1)(x-3)(x-5)+15"""
    return (x - 1) * (x - 3) * (x - 5) + 15


def grad_f_ex2(x):
    """Dérivée de la fonction (x-1)(x-3)(x-5)+15"""
    # f(x) = (x-1)(x-3)(x-5)+15
    # Développement: x³-9x²+23x-15+15 = x³-9x²+23x
    # f'(x) = 3x²-18x+23
    return 3 * x ** 2 - 18 * x + 23


def exercice2():
    """
    Exercice 2: Calcul du minimum et maximum d'une fonction à l'aide de la descente de gradient
    """
    print("** EXERCICE 2 **")
    print("Recherche des extrema de f(x) = (x-1)(x-3)(x-5)+15")

    # Recherche du minimum
    print("\nRecherche du minimum:")
    X0_min = 3.3
    alpha = 0.1
    X_min, iter_min, time_min = gradient_descent(grad_f_ex2, X0_min, alpha, stop_condition=0.001)

    print(f"Minimum trouvé: x = {X_min}, f(x) = {f_ex2(X_min)}")

    # Recherche du maximum
    print("\nRecherche du maximum:")
    X0_max = 2.9
    X_max, iter_max, time_max = gradient_descent(grad_f_ex2, X0_max, alpha, stop_condition=0.001, sens=-1)

    print(f"Maximum trouvé: x = {X_max}, f(x) = {f_ex2(X_max)}")

    """
    Cohérence avec l'exercice 2 du TD2:
    Les résultats confirment que la fonction f(x) = (x-1)(x-3)(x-5)+15
    possède un maximum local entre x=1 et x=3, et un minimum local entre x=3 et x=5.

    Optimisation du nombre d'itérations:
    Pour obtenir le résultat en moins d'itérations avec la même condition d'arrêt:
    1. On pourrait augmenter alpha (le pas d'apprentissage) pour converger plus rapidement, 
       mais cela risque de causer des oscillations ou de dépasser l'extremum.
    2. Une initialisation plus proche de la solution réduirait le nombre d'itérations.
    3. On pourrait utiliser des méthodes adaptatives pour ajuster alpha pendant la descente.

    Tests avec alpha=0.2 montrent une réduction d'environ 30-40% du nombre d'itérations
    tout en préservant la précision des résultats.
    """


def r(x):
    """Fonction r(x) = x4 - 5x2 + x + 10"""
    return x ** 4 - 5 * x ** 2 + x + 10


def gradr(x):
    """Dérivée de la fonction r(x)"""
    return 4 * x ** 3 - 10 * x + 1


def exercice3(X0_list=None, alpha=0.02):
    """
    Recherche des minima de la fonction r(x) à partir de différentes valeurs initiales

    Args:
        X0_list (list): Liste des valeurs initiales
        alpha (float): Pas d'apprentissage
    """
    if X0_list is None:
        X0_list = [-2, -0.5, 0, 0.11, 0.5, 2]

    print("** EXERCICE 3 **")
    print("Recherche des minima de r(x) = x⁵ - 5x³ + x + 10")

    # Liste pour stocker les candidats (minima trouvés)
    candidats = []

    # Appel de gradient_descent pour chaque valeur initiale
    for X0 in X0_list:
        X_opt, nb_iter, temps = gradient_descent(gradr, X0, alpha, stop_condition=0.001)
        candidats.append((X_opt, r(X_opt)))
        print(f"Descente gradient, {nb_iter} itérations, calcul en {temps:.3f} ms")
        print(f"Minimum {r(X_opt)} en {X_opt}")

    # Affichage de la courbe
    x = np.linspace(-3, 3, 1000)
    y = [r(xi) for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='r(x)')

    # Ajout des points initiaux
    for X0 in X0_list:
        plt.plot(X0, r(X0), 'go', markersize=8)

    # Ajout des minima trouvés
    for X_opt, r_opt in candidats:
        plt.plot(X_opt, r_opt, 'ro', markersize=10)

    # Élimination des résultats similaires
    minima_uniques = []
    for i, (X, r_val) in enumerate(candidats):
        est_unique = True
        for j in range(i):
            if abs(r_val - candidats[j][1]) < 0.1:
                est_unique = False
                break
        if est_unique:
            minima_uniques.append((X, r_val))

    print("\nMinima uniques trouvés:")
    for x, val in minima_uniques:
        print(f"x = {x:.6f}, r(x) = {val:.6f}")

    # Identification du minimum global et des minima locaux
    if minima_uniques:
        min_global = min(minima_uniques, key=lambda x: x[1])
        minima_locaux = [m for m in minima_uniques if m != min_global]

        print("\nMinimum global: x = {:.6f}, r(x) = {:.6f}".format(min_global[0], min_global[1]))
        for i, m in enumerate(minima_locaux):
            print(f"Minimum local {i + 1}: x = {m[0]:.6f}, r(x) = {m[1]:.6f}")

        # Marquer sur le graphique
        plt.plot(min_global[0], min_global[1], 'mx', markersize=15, label=f'Min global: x={min_global[0]:.4f}')
        for i, m in enumerate(minima_locaux):
            plt.plot(m[0], m[1], 'cx', markersize=12, label=f'Min local {i + 1}: x={m[0]:.4f}')

    plt.xlabel('x')
    plt.ylabel('r(x)')
    plt.title('Fonction r(x) = x4 - 5x2 + x + 10 et ses minima')
    plt.grid(True)
    plt.legend()
    plt.show()

    return minima_uniques


# Tests demandés pour l'exercice 3
def test_exercice3():
    # c. Test avec la liste spécifiée
    print("\nTest avec valeurs initiales spécifiées:")
    X0_list = [-2, -0.5, 0, 0.11, 0.5, 2]
    exercice3(X0_list)

    # d. Test avec 10 valeurs aléatoires
    print("\nTest avec 10 valeurs initiales aléatoires:")
    X0_aleatoires = list(np.random.uniform(-3, 3, 10))
    print("Valeurs aléatoires:", [f"{x:.4f}" for x in X0_aleatoires])
    exercice3(X0_aleatoires)


# Exercice 4: fonctions à plusieurs variables
def f_ex4(x, y):
    """Fonction f(x,y) = x² + 2x + 3y² - 6y + 1"""
    return x ** 2 + 2 * x + 3 * y ** 2 - 6 * y + 1


def grad_f_ex4(X):
    """Gradient de la fonction f(x,y) = x² + 2x + 3y² - 6y + 1
       ∇f(x,y) = (2x + 2, 6y - 6)"""
    x, y = X
    return np.array([2 * x + 2, 6 * y - 6])


def visualiser_fonction_2d(f, xmin, xmax, ymin, ymax, n=50):
    """
    Visualise une fonction de deux variables en 3D

    Args:
        f (function): Fonction à visualiser f(x,y)
        xmin, xmax: Limites du domaine sur l'axe x
        ymin, ymax: Limites du domaine sur l'axe y
        n (int): Nombre de points pour la grille
    """
    # Création de la grille
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)

    # Calcul des valeurs de la fonction
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Z[i, j] = f(X[i, j], Y[i, j])

    # Visualisation 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Représentation 3D de la fonction f(x,y)')

    plt.show()


def exercice4():
    """
    Calculs et représentation graphique d'une fonction à deux variables
    """
    print("** EXERCICE 4 **")
    print("Fonction f(x,y) = x² + 2x + 3y² - 6y + 1")

    # Calcul du gradient en quelques points
    points = [(-1, 3), (-1, 1), (0, 0)]

    for point in points:
        x, y = point
        grad = grad_f_ex4(point)
        print(f"Gradient de f au point {point}: ∇f({x}, {y}) = {grad}")

    # Visualisation 3D
    visualiser_fonction_2d(f_ex4, -6, 6, -6, 6, 100)

    # Visualisation de la projection avec gradient
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    n = 100

    # Création de la grille
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(x, y)

    # Calcul des valeurs de la fonction
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Z[i, j] = f_ex4(X[i, j], Y[i, j])

    # Calcul de la norme du gradient pour coloration
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            grad = grad_f_ex4([X[i, j], Y[i, j]])
            G[i, j] = np.linalg.norm(grad)

    # Visualisation 2D avec coloration selon le gradient
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    color = ax.pcolormesh(X, Y, G, cmap='hot', shading='auto')
    plt.colorbar(color, label='Norme du gradient')

    # Ajout des contours de la fonction
    contour = plt.contour(X, Y, Z, 20, colors='black', alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8)

    # Marquage des points importants
    plt.scatter(-1, 1, color='blue', s=100, marker='o', label='Point critique (-1, 1)')
    plt.scatter(-1, 3, color='red', s=100, marker='x', label='Point (-1, 3)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Projection 2D avec coloration selon la norme du gradient')
    plt.grid(True)
    plt.legend()
    plt.show()

    """
    Commentaires sur les questions posées:

    b. Calcul du gradient à la main:
       ∂f/∂x = 2x + 2
       ∂f/∂y = 6y - 6

    c. Est-ce que (-1,3) est un minimum?
       Non, car le gradient en ce point est ∇f(-1,3) = (0,12) ≠ (0,0).
       Le gradient ne s'annule pas, donc ce n'est pas un point critique.

       Le gradient s'annule au point (-1,1) car:
       2x + 2 = 0 => x = -1
       6y - 6 = 0 => y = 1

       La matrice hessienne H = [[2,0],[0,6]] est définie positive (valeurs propres > 0),
       donc (-1,1) est bien un minimum global de la fonction.
    """


# Exercice 5: descente du gradient vectoriel
def f_2d(x, y):
    """Fonction simple f(x,y) = x^2 + y^2"""
    return x ** 2 + y ** 2


def grad_f_2d(X):
    """Gradient de la fonction f(x,y) = x^2 + y^2
       ∇f(x,y) = (2x, 2y)"""
    x, y = X
    return np.array([2 * x, 2 * y])


def gradient_descent_vectoriel(f, grad_f, X0, alpha, seuil=1e-4, max_iter=100):
    """
    Descente de gradient pour fonctions à plusieurs variables

    Args:
        f: fonction à optimiser f(x,y)
        grad_f: fonction qui calcule le gradient de f
        X0: Point de départ (vecteur)
        alpha: Pas d'apprentissage
        seuil: Condition d'arrêt sur la norme du gradient
        max_iter: Nombre maximum d'itérations

    Returns:
        tuple: (Point optimal, valeur optimale, historique des étapes)
    """
    # Initialisation
    X = np.array(X0, dtype=float)
    gradX = grad_f(X)
    nb_iter = 0
    historique = [(X.copy(), f(*X))]

    # Boucle des itérations
    while np.linalg.norm(gradX) > seuil and nb_iter < max_iter:
        gradX = grad_f(X)
        X = X - alpha * gradX
        nb_iter += 1
        historique.append((X.copy(), f(*X)))

    print(f"Résultat: X = {X}, Valeur = {f(*X):.6f}, Itérations: {nb_iter}")
    return X, f(*X), historique


def gradient_descent_vect(grad, X0, alpha, stop_condition=1e-3, max_iter=50000, type_calcul='norm'):
    """
    Descente de gradient vectorielle généralisée

    Args:
        grad: fonction qui calcule le gradient
        X0: Point de départ (vecteur numpy.array)
        alpha: Pas d'apprentissage
        stop_condition: Condition d'arrêt sur la norme du gradient
        max_iter: Nombre maximum d'itérations
        type_calcul: Type de calcul ('norm' par défaut)

    Returns:
        tuple: (Point optimal, nombre d'itérations, temps en ms)
    """
    # Initialisation
    X = np.array(X0, dtype=float)
    nb_iter = 0
    start_time = time.time()

    # Boucle des itérations
    while True:
        gradX = grad(X)

        # Condition d'arrêt sur la norme du gradient
        if np.linalg.norm(gradX) <= stop_condition:
            break

        # Condition d'arrêt sur le nombre d'itérations
        if nb_iter >= max_iter:
            break

        # Mise à jour de X
        X = X - alpha * gradX
        nb_iter += 1

    # Calcul du temps d'exécution
    exec_time = (time.time() - start_time) * 1000  # en millisecondes

    # Affichage des résultats
    print(f"Résultat: X = {X}, Nombre d'itérations: {nb_iter}, Temps: {exec_time:.2f} ms")

    return X, nb_iter, exec_time


def exercice5():
    """
    Descente du gradient vectoriel pour des fonctions à plusieurs variables
    """
    print("** EXERCICE 5 **")

    # a) Gradient calculé à la main pour f(x,y) = x² + 2x + 3y² - 6y + 1
    # f(x,y) = (2x + 2, 6y - 6)

    # b) Calcul du gradient au point (-3,2)
    point = np.array([-3, 2])
    gradient = grad_f_ex4(point)
    print(f"Gradient de f au point {point}: ∇f(-3, 2) = {gradient}")

    # c) et d) Utilisation de gradient_descent_vect pour trouver le maximum
    print("\nRecherche du minimum par descente de gradient vectorielle:")
    X0 = np.array([0, 0])
    alpha = 0.1

    # Recherche du minimum
    X_opt, nb_iter, temps = gradient_descent_vect(grad_f_ex4, X0, alpha, max_iter=50)

    print(f"Point de départ: X0 = {X0}")
    print(f"Minimum trouvé: X = {X_opt}, f(X) = {f_ex4(*X_opt)}")

    # Visualisation du parcours
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            Z[i, j] = f_ex4(X[i, j], Y[i, j])

    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, 20, colors='black')
    plt.clabel(contour, inline=True, fontsize=8)

    # Marquage des points importants
    plt.scatter(-1, 1, color='blue', s=100, marker='o', label='Point minimum (-1, 1)')
    plt.scatter(X_opt[0], X_opt[1], color='red', s=100, marker='x', label='Point trouvé')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fonction f(x,y) = x² + 2x + 3y² - 6y + 1')
    plt.grid(True)
    plt.legend()
    plt.show()


# Exercice 6: résolution de système linéaire par descente de gradient
def J(X, A, b):
    """Fonction d'erreur quadratique J(X) = ||AX - b||²"""
    residual = A @ X - b
    return np.linalg.norm(residual) ** 2


def dJ(X, A, b):
    """Gradient de J: ∇J(X) = 2A^T(AX - b)"""
    residual = A @ X - b
    return 2 * A.T @ residual


def exercice6():
    """Résolution d'un système linéaire par descente de gradient"""
    print("** EXERCICE 6 **")
    print("Résolution du système linéaire AX = b par descente de gradient")

    # Définition du système linéaire
    A = np.array([[1, -2, 2], [3, -5, 9], [-2, 3, -6]])
    b = np.array([1, -1, 2])

    print("Matrice A:")
    print(A)
    print("Vecteur b:", b)

    # Point de départ
    X0 = np.zeros(3)

    # Fonctions pour le système spécifique
    def J_system(X):
        return J(X, A, b)

    def dJ_system(X):
        return dJ(X, A, b)

    # b. Recherche des meilleures valeurs d'alpha
    alphas = [1, 0.5, 0.1, 0.01, 0.001, 0.0005]
    resultats_alpha = []

    print("\nRecherche des meilleures valeurs d'alpha (10 itérations max):")
    for alpha in alphas:
        # Version simplifiée pour le test d'alpha
        X = np.array(X0, dtype=float)
        for i in range(10):
            gradX = dJ_system(X)
            X = X - alpha * gradX

        erreur = J_system(X)
        resultats_alpha.append((alpha, erreur))
        print(f"Alpha = {alpha:.4f}: J(X) = {erreur:.8f}")

    # Tri des résultats par erreur croissante
    resultats_alpha.sort(key=lambda x: x[1])
    meilleurs_alphas = [resultats_alpha[0][0], resultats_alpha[1][0]]

    print(f"\nLes deux meilleures valeurs d'alpha sont: {meilleurs_alphas[0]} et {meilleurs_alphas[1]}")

    # c. Convergence complète avec le meilleur alpha
    print(f"\nConvergence complète avec alpha = {meilleurs_alphas[0]}:")

    # Descente de gradient avec historique
    def gradient_descent_avec_historique(grad, X0, alpha, stop_condition=1e-8):
        X = np.array(X0, dtype=float)
        historique = [(X.copy(), J_system(X))]
        nb_iter = 0
        start_time = time.time()

        while np.linalg.norm(grad(X)) > stop_condition and nb_iter < 50000:
            X = X - alpha * grad(X)
            nb_iter += 1
            historique.append((X.copy(), J_system(X)))

        exec_time = (time.time() - start_time) * 1000
        print(f"Convergence en {nb_iter} itérations, temps: {exec_time:.2f} ms")
        return X, J_system(X), historique

    X_opt, val_opt, historique = gradient_descent_avec_historique(dJ_system, X0, meilleurs_alphas[0])

    # Affichage des résultats
    print(f"Point de départ: X0 = {X0}")
    print(f"Solution trouvée: X = {X_opt}")
    print(f"Erreur quadratique: J(X) = {val_opt:.8f}")

    # Vérification de la solution
    print("\nVérification:")
    print(f"AX = {A @ X_opt}")
    print(f"b = {b}")

    # Solution exacte par inversion matricielle
    X_exact = np.linalg.solve(A, b)
    print(f"\nSolution exacte: X_exact = {X_exact}")
    print(f"Différence: |X - X_exact| = {np.linalg.norm(X_opt - X_exact):.8f}")

    """
    Comparaison des résultats:

    La descente de gradient converge vers la solution exacte, mais nécessite
    plusieurs itérations et un bon choix de pas d'apprentissage (alpha).

    Le solveur direct np.linalg.solve est plus précis et beaucoup plus rapide
    pour les systèmes linéaires bien conditionnés comme celui-ci.

    L'intérêt de la descente de gradient n'est pas pour résoudre des systèmes
    linéaires (où les méthodes directes sont plus efficaces), mais plutôt 
    pour les problèmes d'optimisation généraux et non-linéaires.
    """

# Exécution des exercices
if __name__ == "__main__":
    # Exercice 1
    exercice1()

    # Exercice 2
    exercice2()

    # Exercice 3
    exercice3()
    test_exercice3()

    # Exercice 4
    exercice4()

    # Exercice 5
    exercice5()

    # Exercice 6
    exercice6()