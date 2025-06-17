import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random


def set_data():
    """
    Fonction qui prépare les données du jeu de données Iris.

    Returns:
        X: Données des caractéristiques (features)
        y: Étiquettes des variétés (target)
        xlabels: Noms des caractéristiques
        X_pred: Données de prédiction (1/5 du jeu de données)
        y_pred: Étiquettes correspondant aux données de prédiction
        X_train: Données d'entraînement (4/5 du jeu de données)
        y_train: Étiquettes correspondant aux données d'entraînement
    """
    # Chargement du jeu de données Iris
    iris = datasets.load_iris()

    # Récupération des données et des étiquettes
    X = iris.data
    y = iris.target

    # Noms des caractéristiques mesurées
    xlabels = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # Division des données en ensembles de prédiction et d'entraînement
    # Nous prenons 1/5 des données pour la prédiction et 4/5 pour l'entraînement
    n_samples = len(X)
    n_pred = n_samples // 5  # 1/5 des données

    # Indices aléatoires pour la division des données
    indices = np.random.permutation(n_samples)
    pred_indices = indices[:n_pred]
    train_indices = indices[n_pred:]

    # Création des ensembles de prédiction et d'entraînement
    X_pred = X[pred_indices]
    y_pred = y[pred_indices]
    X_train = X[train_indices]
    y_train = y[train_indices]

    return X, y, xlabels, X_pred, y_pred, X_train, y_train


def exercice1b():
    """
    Représentation graphique des 4 caractéristiques en fonction de la variété d'iris.
    """
    # Récupération des données
    X, y, xlabels, _, _, _, _ = set_data()

    # Création de la figure avec 4 sous-graphiques (un par caractéristique)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))

    # Couleurs pour chaque variété
    colors = ['blue', 'yellow', 'green', 'red']

    # Pour chaque caractéristique
    for i in range(4):
        # Création d'un scatter plot pour chaque variété
        for variety in range(3):
            # Extraction des valeurs de la caractéristique i pour la variété actuelle
            variety_data = X[y == variety, i]

            # Création d'un array de positions x (toutes à la position de la variété)
            x_positions = np.ones_like(variety_data) * variety

            # Traçage des points avec la couleur correspondante
            axs[i].scatter(x_positions, variety_data, c=colors[i], alpha=0.8)

        # Configuration des axes et titres
        axs[i].set_xlabel('variety')
        axs[i].set_ylabel(xlabels[i])
        axs[i].set_xticks([0, 1, 2])

        # Ajustement des limites des axes y
        if i == 0:  # sepal length
            axs[i].set_ylim(4.0, 8.0)
        elif i == 1:  # sepal width
            axs[i].set_ylim(2.0, 4.5)
        elif i == 2:  # petal length
            axs[i].set_ylim(1.0, 7.0)
        elif i == 3:  # petal width
            axs[i].set_ylim(0.0, 2.5)

    # Ajustement de la mise en page
    plt.tight_layout()
    plt.show()




def split(X, y, threshold, feature_idx):
    """
    Divise un jeu de données en deux sous-ensembles basés sur un seuil
    pour une caractéristique donnée.

    Args:
        X: Array de caractéristiques de forme (n_samples, n_features)
        y: Array d'étiquettes de forme (n_samples,)
        threshold: Seuil pour la division
        feature_idx: Indice de la caractéristique à utiliser pour la division

    Returns:
        X1, y1, X2, y2: Les deux sous-ensembles résultants
    """
    # Création de masques pour les conditions
    mask_below = X[:, feature_idx] <= threshold
    mask_above = ~mask_below  # Négation du masque précédent

    # Division des données
    X1 = X[mask_below]
    y1 = y[mask_below]
    X2 = X[mask_above]
    y2 = y[mask_above]

    return X1, y1, X2, y2


def exercice2b():
    """
    Teste la fonction split sur le jeu de données Iris.
    """
    # Chargement du jeu de données
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Application de split avec un seuil de 5.2 sur la première caractéristique (indice 0)
    X1, y1, X2, y2 = split(X, y, threshold=5.2, feature_idx=0)

    # Affichage des longueurs des vecteurs résultants
    print(f"Longueur de X1: {len(X1)}")
    print(f"Longueur de X2: {len(X2)}")


# Exécution de la fonction
exercice2b()


def entropy(y):
    """
    Calcule l'entropie d'un ensemble d'étiquettes.

    Args:
        y: Array d'étiquettes

    Returns:
        L'entropie de l'ensemble
    """
    # Si l'ensemble est vide, l'entropie est 0
    if len(y) == 0:
        return 0

    # Calcul des proportions de chaque classe
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)

    # Calcul de l'entropie
    entropy_value = -np.sum(proportions * np.log2(proportions))
    return entropy_value


def information_gain(y, y1, y2):
    """
    Calcule le gain d'information après une division.

    Args:
        y: Étiquettes de l'ensemble original
        y1: Étiquettes du premier sous-ensemble
        y2: Étiquettes du deuxième sous-ensemble

    Returns:
        Le gain d'information
    """
    # Entropie de l'ensemble original
    entropy_parent = entropy(y)

    # Calcul de l'entropie pondérée des sous-ensembles
    n = len(y)
    n1 = len(y1)
    n2 = len(y2)

    entropy_children = (n1 / n) * entropy(y1) + (n2 / n) * entropy(y2)

    # Gain d'information
    return entropy_parent - entropy_children


def find_best_split(X, y, feature_idx):
    """
    Trouve le meilleur seuil pour diviser les données selon une caractéristique donnée.

    Args:
        X: Array de caractéristiques
        y: Array d'étiquettes
        feature_idx: Indice de la caractéristique à utiliser

    Returns:
        Le meilleur seuil et le gain d'information correspondant
    """
    # Extraction des valeurs uniques pour la caractéristique
    values = np.sort(np.unique(X[:, feature_idx]))

    best_gain = -1
    best_threshold = None

    # Essai de tous les seuils possibles
    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2
        X1, y1, X2, y2 = split(X, y, threshold, feature_idx)

        # Calcul du gain d'information
        gain = information_gain(y, y1, y2)

        # Mise à jour du meilleur seuil
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def find_best_feature_and_split(X, y):
    """
    Trouve la meilleure caractéristique et le meilleur seuil pour diviser les données.

    Args:
        X: Array de caractéristiques
        y: Array d'étiquettes

    Returns:
        Le meilleur indice de caractéristique, le meilleur seuil et le gain correspondant
    """
    n_features = X.shape[1]
    best_feature = None
    best_threshold = None
    best_gain = -1

    # Essai de toutes les caractéristiques
    for feature_idx in range(n_features):
        threshold, gain = find_best_split(X, y, feature_idx)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_feature = feature_idx

    return best_feature, best_threshold, best_gain


def exercice2c():
    """
    Teste la fonction find_best_split sur le jeu de données Iris.
    """
    # Chargement du jeu de données
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Trouver le meilleur seuil pour la caractéristique 0
    best_threshold, best_gain = find_best_split(X, y, 0)

    print(f"Pour la caractéristique 0:")
    print(f"  Meilleur seuil: {best_threshold:.2f}")
    print(f"  Gain d'information: {best_gain:.4f}")


def exercice2d():
    """
    Teste la fonction find_best_feature_and_split sur le jeu de données Iris.
    """
    # Chargement du jeu de données
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Trouver la meilleure caractéristique et le meilleur seuil
    best_feature, best_threshold, best_gain = find_best_feature_and_split(X, y)

    print(f"Meilleure caractéristique: {best_feature}")
    print(f"Meilleur seuil: {best_threshold:.2f}")
    print(f"Gain d'information: {best_gain:.4f}")


# Exécution des fonctions
exercice2c()
exercice2d()


class DecisionTreeNode:
    """
    Classe représentant un nœud dans un arbre de décision.
    """

    def __init__(self):
        self.feature_idx = None  # Caractéristique à tester
        self.threshold = None  # Seuil pour la division
        self.left = None  # Nœud enfant gauche (condition ≤ threshold)
        self.right = None  # Nœud enfant droit (condition > threshold)
        self.is_leaf = False  # Indique si le nœud est une feuille
        self.prediction = None  # Classe prédite (pour les feuilles)


def build_tree(X, y, max_depth=None, min_samples_split=2, depth=0):
    """
    Construit un arbre de décision récursivement.

    Args:
        X: Array de caractéristiques
        y: Array d'étiquettes
        max_depth: Profondeur maximale de l'arbre
        min_samples_split: Nombre minimum d'échantillons pour diviser un nœud
        depth: Profondeur actuelle dans l'arbre

    Returns:
        Un objet DecisionTreeNode représentant l'arbre ou le sous-arbre
    """
    node = DecisionTreeNode()

    # Vérifier les conditions d'arrêt
    if (max_depth is not None and depth >= max_depth) or len(X) < min_samples_split or len(np.unique(y)) == 1:
        node.is_leaf = True
        # Prédiction basée sur la classe majoritaire
        unique_classes, counts = np.unique(y, return_counts=True)
        node.prediction = unique_classes[np.argmax(counts)]
        return node

    # Trouver la meilleure division
    best_feature, best_threshold, best_gain = find_best_feature_and_split(X, y)

    # Si aucune division n'améliore l'entropie
    if best_gain <= 0:
        node.is_leaf = True
        unique_classes, counts = np.unique(y, return_counts=True)
        node.prediction = unique_classes[np.argmax(counts)]
        return node

    # Configurer le nœud
    node.feature_idx = best_feature
    node.threshold = best_threshold

    # Diviser les données
    X1, y1, X2, y2 = split(X, y, best_threshold, best_feature)

    # Construire récursivement les sous-arbres
    node.left = build_tree(X1, y1, max_depth, min_samples_split, depth + 1)
    node.right = build_tree(X2, y2, max_depth, min_samples_split, depth + 1)

    return node


def predict_single(node, x):
    """
    Prédit la classe d'une seule observation en parcourant l'arbre.

    Args:
        node: Nœud actuel de l'arbre
        x: Observation à classer

    Returns:
        La classe prédite
    """
    if node.is_leaf:
        return node.prediction

    if x[node.feature_idx] <= node.threshold:
        return predict_single(node.left, x)
    else:
        return predict_single(node.right, x)


def predict(node, X):
    """
    Prédit les classes d'un ensemble d'observations.

    Args:
        node: Racine de l'arbre de décision
        X: Ensemble d'observations à classer

    Returns:
        Array des classes prédites
    """
    return np.array([predict_single(node, x) for x in X])


def exercice2e():
    """
    Construit un arbre de décision complet pour le jeu de données Iris
    et évalue ses performances.
    """
    # Chargement du jeu de données
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Construction de l'arbre
    tree_root = build_tree(X, y, max_depth=3)

    # Prédiction sur l'ensemble d'entraînement
    y_pred = predict(tree_root, X)

    # Calcul de la précision
    accuracy = np.mean(y_pred == y)
    print(f"Précision de notre arbre de décision: {accuracy:.4f}")


# Exécution de la fonction
exercice2e()

from sklearn import tree, datasets
import matplotlib.pyplot as plt


def exercice3():
    """
    Construit et visualise un arbre de décision avec scikit-learn.
    """
    # Chargement du jeu de données
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Phase d'apprentissage
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)

    # Affichage de l'arbre
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
    plt.title("Arbre de décision pour les données Iris avec scikit-learn")
    plt.tight_layout()
    plt.show()

    # Prédiction pour un nouvel échantillon
    nouvel_echantillon = [[6, 3, 4, 1]]
    prediction = clf.predict(nouvel_echantillon)
    print(f"Prédiction pour l'échantillon {nouvel_echantillon}: {iris.target_names[prediction[0]]}")


# Exécution de la fonction
class Node:
    """
    Classe représentant un nœud dans un arbre de décision simplifié.

    Attributes:
        variable: Indice de la variable à tester (0-3), None pour les feuilles
        seuil: Valeur seuil pour le test, None pour les feuilles
        variete: Variété de la feuille, None pour les nœuds
        gauche: Nœud enfant gauche (condition <= seuil)
        droite: Nœud enfant droit (condition > seuil)
    """

    def __init__(self, variable, seuil, variete, gauche=None, droite=None):
        """
        Initialise un nœud de l'arbre de décision.

        Args:
            variable: Indice de la variable à tester (0-3), None pour les feuilles
            seuil: Valeur seuil pour le test, None pour les feuilles
            variete: Variété de la feuille, None pour les nœuds
            gauche: Nœud enfant gauche (condition <= seuil)
            droite: Nœud enfant droit (condition > seuil)
        """
        self.variable = variable
        self.seuil = seuil
        self.variete = variete
        self.gauche = gauche
        self.droite = droite

    def predict(self, data):
        """
        Prédit la variété d'une fleur en parcourant l'arbre.

        Args:
            data: Vecteur de mesures [v0, v1, v2, v3]

        Returns:
            La variété prédite
        """
        # Si c'est une feuille (variable et seuil sont None)
        if self.variable is None and self.seuil is None:
            return self.variete

        # Sinon, c'est un nœud de test
        if data[self.variable] <= self.seuil:
            return self.gauche.predict(data)
        else:
            return self.droite.predict(data)


def exercice1c():
    """
    Construit un arbre de décision simplifié pour le jeu de données Iris
    et évalue ses performances.
    """
    # Récupération des données
    X, y, _, X_pre, y_pre, _, _ = set_data()

    # Création des feuilles
    feuille_variete0 = Node(None, None, [0], None, None)
    feuille_variete12_1 = Node(None, None, [1, 2], None, None)
    feuille_variete12_2 = Node(None, None, [1, 2], None, None)

    # Création des nœuds
    noeud_v3 = Node(3, 0.85, None, feuille_variete0, feuille_variete12_1)
    racine = Node(2, 2.7, None, noeud_v3, feuille_variete12_2)

    # Test de prédiction pour 3 variétés différentes
    print(f"Donnee : {X_pre[0]} variété : {y_pre[0]}")
    print(f"Prediction : {racine.predict(X_pre[0])}")

    print(f"Donnee : {X_pre[10]} variété : {y_pre[10]}")
    print(f"Prediction : {racine.predict(X_pre[10])}")

    print(f"Donnee : {X_pre[20]} variété : {y_pre[20]}")
    print(f"Prediction : {racine.predict(X_pre[20])}")

    # Calcul du taux de prédictions correctes
    correct = 0
    total = len(X_pre)

    for i in range(total):
        prediction = racine.predict(X_pre[i])
        # Une prédiction est correcte si une seule réponse est renvoyée et si elle correspond à la variété
        if isinstance(prediction, list) and len(prediction) == 1 and prediction[0] == y_pre[i]:
            correct += 1
        elif not isinstance(prediction, list) and prediction == y_pre[i]:
            correct += 1
        elif isinstance(prediction, list) and y_pre[i] in prediction:
            # Si la prédiction est [1, 2] et la variété est 1 ou 2, c'est partiellement correct
            # mais pas compté comme correct selon l'énoncé
            pass

    print(f"Taux de prédiction correcte : {correct/total:.4f}")

    # Test avec la donnée [7.5, 3.8, 2.2, 0.7]
    test_data = [7.5, 3.8, 2.2, 0.7]
    print(f"Donnee : {test_data}")
    print(f"Prediction : {racine.predict(test_data)}")


def exercice1d():
    """
    Affiche deux nuages de points pour visualiser la séparation des variétés.
    """
    # Récupération des données
    X, y, _, _, _, _, _ = set_data()

    # Création de la figure avec 2 sous-graphiques
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Couleurs pour chaque variété
    colors = ['red', 'green', 'blue']

    # Premier graphique : longueur sépale vs largeur sépale
    for variety in range(3):
        # Extraction des données pour la variété actuelle
        variety_data = X[y == variety]
        axs[0].scatter(variety_data[:, 0], variety_data[:, 1], c=colors[variety], label=f'Variété {variety}')

    axs[0].set_xlabel('Longueur sépale (cm)')
    axs[0].set_ylabel('Largeur sépale (cm)')
    axs[0].set_title('Longueur sépale vs Largeur sépale')
    axs[0].legend()

    # Deuxième graphique : longueur pétale vs largeur pétale
    for variety in range(3):
        # Extraction des données pour la variété actuelle
        variety_data = X[y == variety]
        axs[1].scatter(variety_data[:, 2], variety_data[:, 3], c=colors[variety], label=f'Variété {variety}')

    axs[1].set_xlabel('Longueur pétale (cm)')
    axs[1].set_ylabel('Largeur pétale (cm)')
    axs[1].set_title('Longueur pétale vs Largeur pétale')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # Commentaire sur la séparation des variétés
    print("Commentaire : D'après les graphiques, on peut voir que la variété 0 est facilement séparable des variétés 1 et 2")
    print("en utilisant les mesures de pétales (longueur et largeur). Cependant, il n'est pas possible de déterminer")
    print("des critères simples de séparation pour distinguer les variétés 1 et 2 car leurs distributions se chevauchent.")


def gini(y):
    """
    Calcule le coefficient de Gini pour un ensemble d'étiquettes.

    Args:
        y: Array d'étiquettes

    Returns:
        Le coefficient de Gini
    """
    # Si l'ensemble est vide, le coefficient de Gini est 0
    if len(y) == 0:
        return 0

    # Calcul des proportions de chaque classe (0, 1, 2)
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)

    # Calcul du coefficient de Gini
    gini_value = sum(p * (1 - p) for p in proportions)
    return gini_value


def exercice2a():
    """
    Calcule et affiche le coefficient de Gini pour le jeu de données Iris.
    """
    # Récupération des données
    _, y, _, _, _, _, _ = set_data()

    # Calcul du coefficient de Gini
    gini_value = gini(y)

    print(f"gini y initial : {gini_value}")
    return gini_value


def split_opt(X, y, k):
    """
    Trouve le seuil optimal pour diviser les données selon une caractéristique donnée,
    en minimisant le coefficient de Gini.

    Args:
        X: Array de caractéristiques
        y: Array d'étiquettes
        k: Indice de la caractéristique à utiliser

    Returns:
        Le meilleur seuil et le gain sur le coefficient de Gini
    """
    # Calcul du coefficient de Gini initial
    initial_gini = gini(y)

    # Extraction des valeurs pour la caractéristique k
    feature_values = X[:, k]

    # Détermination de l'intervalle de recherche
    min_value = np.min(feature_values)
    max_value = np.max(feature_values)

    best_gini = float('inf')
    best_threshold = None

    # Essai de tous les seuils possibles avec un pas de 0.1
    for threshold in np.arange(min_value, max_value + 0.1, 0.1):
        # Division des données
        X1, y1, X2, y2 = split(X, y, threshold, k)

        # Calcul des coefficients de Gini pour chaque sous-ensemble
        gini1 = gini(y1)
        gini2 = gini(y2)

        # Calcul de la métrique combinée (moyenne pondérée)
        p1 = len(y1) / len(y)
        p2 = len(y2) / len(y)
        combined_gini = p1 * gini1 + p2 * gini2

        # Mise à jour du meilleur seuil
        if combined_gini < best_gini:
            best_gini = combined_gini
            best_threshold = threshold

    # Calcul du gain
    gain = initial_gini - best_gini

    return best_threshold, gain, gini(y1), gini(y2)


def exercice2c():
    """
    Teste la fonction split_opt sur le jeu de données Iris.
    """
    # Récupération des données
    X, y, _, _, _, _, _ = set_data()

    # Trouver le meilleur seuil pour la caractéristique 3
    best_threshold, gain, gini1, gini2 = split_opt(X, y, 3)

    print(f"Variable 3 : seuil optimal {best_threshold} gain optimal {gain:.4f}")
    print(f"Coefficient de Gini du groupe 1 = {gini1:.4f}")
    print(f"Coefficient de Gini du groupe 2 = {gini2:.4f}")

    return best_threshold, gain, gini1, gini2


def K_opt(X, y):
    """
    Trouve la meilleure caractéristique et le meilleur seuil pour diviser les données,
    en minimisant le coefficient de Gini.

    Args:
        X: Array de caractéristiques
        y: Array d'étiquettes

    Returns:
        Le meilleur indice de caractéristique, le meilleur seuil et le gain correspondant
    """
    best_k = None
    best_threshold = None
    best_gain = -1

    # Essai de toutes les caractéristiques (0, 1, 2, 3)
    for k in range(X.shape[1]):
        threshold, gain, _, _ = split_opt(X, y, k)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
            best_k = k

    return best_k, best_threshold, best_gain


def exercice2d():
    """
    Teste la fonction K_opt sur le jeu de données Iris.
    """
    # Récupération des données
    X, y, _, _, _, _, _ = set_data()

    # Trouver la meilleure caractéristique et le meilleur seuil
    best_k, best_threshold, best_gain = K_opt(X, y)

    print(f"Critère de séparation optimal k = {best_k}, s = {best_threshold}")
    print(f"Gain en pureté = {best_gain:.4f}")

    # Séparation des données selon le meilleur critère
    X1, y1, X2, y2 = split(X, y, best_threshold, best_k)

    # Calcul des coefficients de Gini pour chaque sous-groupe
    gini1 = gini(y1)
    gini2 = gini(y2)

    print(f"Coefficient de Gini du groupe 1 = {gini1:.4f}")
    print(f"Coefficient de Gini du groupe 2 = {gini2:.4f}")


class Node2:
    """
    Classe représentant un nœud dans un arbre de décision automatique.

    Attributes:
        X: Données des caractéristiques
        y: Étiquettes des variétés
        k: Indice de la variable à tester
        s: Seuil pour le test
        left: Nœud enfant gauche
        right: Nœud enfant droit
    """

    def __init__(self, X, y):
        """
        Initialise un nœud de l'arbre de décision.

        Args:
            X: Données des caractéristiques
            y: Étiquettes des variétés
        """
        self.X = X
        self.y = y
        self.k = None
        self.s = None
        self.left = None
        self.right = None

    def leaves(self):
        """
        Retourne la liste des feuilles de l'arbre.

        Returns:
            Liste des feuilles
        """
        if self.k is None:  # C'est une feuille
            return [self]

        # Sinon, c'est un nœud interne
        return self.left.leaves() + self.right.leaves()

    def grow(self, pruning=0):
        """
        Fait croître l'arbre en séparant la feuille avec le meilleur gain.

        Args:
            pruning: Nombre minimum d'échantillons pour autoriser une séparation

        Returns:
            True si un gain en pureté a été réalisé, False sinon
        """
        # Si ce n'est pas une feuille, on ne peut pas la faire croître
        if self.k is not None:
            return False

        # Récupération de toutes les feuilles
        leaves = self.leaves()

        best_gain = -1
        best_leaf = None
        best_k = None
        best_s = None

        # Recherche de la feuille avec le meilleur gain
        for leaf in leaves:
            if len(leaf.X) <= 1:  # Pas assez de données pour diviser
                continue

            k, s, gain = K_opt(leaf.X, leaf.y)

            if gain > best_gain:
                best_gain = gain
                best_leaf = leaf
                best_k = k
                best_s = s

        # Si aucun gain n'est possible ou si le gain est nul
        if best_gain <= 0 or best_leaf is None:
            return False

        # Division des données
        X1, y1, X2, y2 = split(best_leaf.X, best_leaf.y, best_s, best_k)

        # Vérification de la contrainte d'élagage
        if pruning > 0 and (len(X1) <= pruning or len(X2) <= pruning):
            return False

        # Création des nœuds enfants
        best_leaf.k = best_k
        best_leaf.s = best_s
        best_leaf.left = Node2(X1, y1)
        best_leaf.right = Node2(X2, y2)

        return True

    def extend(self, pruning=0):
        """
        Étend l'arbre autant que possible.

        Args:
            pruning: Nombre minimum d'échantillons pour autoriser une séparation
        """
        while self.grow(pruning):
            pass

    def print(self, indent=0):
        """
        Affiche l'arbre.

        Args:
            indent: Niveau d'indentation
        """
        if self.k is None:  # C'est une feuille
            print(" " * indent + f"taille = {len(self.y)}")
            print(" " * indent + f"gini = {gini(self.y):.4f}")
        else:  # C'est un nœud interne
            print(" " * indent + f"k = {self.k}, s = {self.s}")
            print(" " * indent + "L :")
            self.left.print(indent + 2)
            print(" " * indent + "R :")
            self.right.print(indent + 2)

    def predict(self, x):
        """
        Prédit la variété d'une fleur en parcourant l'arbre.

        Args:
            x: Vecteur de mesures [v0, v1, v2, v3]

        Returns:
            La variété prédite
        """
        if self.k is None:  # C'est une feuille
            if len(np.unique(self.y)) == 1:  # Feuille pure
                return self.y[0]
            else:  # Feuille impure, on tire au hasard
                return random.choice(self.y)

        # Sinon, c'est un nœud de test
        if x[self.k] <= self.s:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


def exercice2e():
    """
    Construit un arbre de décision automatique pour le jeu de données Iris
    et affiche sa structure.
    """
    # Récupération des données
    X, y, _, _, _, _, _ = set_data()

    # Création de la racine
    root = Node2(X, y)

    # Premier appel à grow
    root.grow(pruning=0)

    # Affichage des informations de la racine
    print(f"Séparation pour k: {root.k} s: {root.s}")
    print(f"Coefficient de Gini gauche : {gini(root.left.y):.4f}")
    print(f"Coefficient de Gini droit : {gini(root.right.y):.4f}")

    # Affichage de l'arbre
    print("Arbre :")
    root.print()


def exercice2f():
    """
    Construit un arbre de décision automatique complet pour le jeu de données Iris
    et affiche sa structure.
    """
    # Récupération des données
    X, y, _, _, _, _, _ = set_data()

    # Création de la racine
    root = Node2(X, y)

    # Premier appel à grow puis extend
    root.grow(pruning=0)
    root.extend(pruning=0)

    # Affichage de l'arbre
    print("Arbre complet :")
    root.print()


def exercice2g():
    """
    Construit un arbre de décision automatique pour le jeu de données Iris
    et évalue ses performances.
    """
    # Récupération des données
    X, y, _, X_pre, y_pre, X_train, y_train = set_data()

    # Création de la racine avec les données d'entraînement
    root = Node2(X, y)

    # Construction de l'arbre
    root.grow(pruning=0)
    root.extend(pruning=0)

    # Test de prédiction pour 3 variétés différentes
    print(f"Donnee : {X_pre[0]} variété : {y_pre[0]}")
    print(f"Prediction : {root.predict(X_pre[0])}")

    print(f"Donnee : {X_pre[10]} variété : {y_pre[10]}")
    print(f"Prediction : {root.predict(X_pre[10])}")

    print(f"Donnee : {X_pre[20]} variété : {y_pre[20]}")
    print(f"Prediction : {root.predict(X_pre[20])}")

    # Test avec la donnée [7.5, 3.8, 2.2, 0.7]
    test_data = [7.5, 3.8, 2.2, 0.7]
    print(f"Donnee : {test_data}")
    print(f"Prediction : {root.predict(test_data)}")

    # Calcul du taux de prédictions correctes
    correct = 0
    total = len(X_train)

    for i in range(total):
        prediction = root.predict(X_train[i])
        if prediction == y_train[i]:
            correct += 1

    print(f"Taux de prédiction correcte sur les données d'entraînement : {correct/total:.4f}")

    # Commentaire sur la cohérence avec l'exercice 1c
    print("Commentaire : Les résultats sont cohérents avec ceux de l'exercice 1c.")
    print("L'arbre automatique est plus précis car il peut distinguer les variétés 1 et 2.")


def exercice2h():
    """
    Construit un arbre de décision automatique avec les données d'entraînement
    et évalue ses performances sur les données de prédiction.
    """
    # Récupération des données
    _, _, _, X_pre, y_pre, X_train, y_train = set_data()

    # Création de la racine avec les données d'entraînement
    root = Node2(X_train, y_train)

    # Construction de l'arbre
    root.grow(pruning=0)
    root.extend(pruning=0)

    # Calcul du taux de prédictions correctes sur les données de prédiction
    correct = 0
    total = len(X_pre)

    for i in range(total):
        prediction = root.predict(X_pre[i])
        if prediction == y_pre[i]:
            correct += 1

    print(f"Taux de prédiction correcte sur les données de prédiction : {correct/total:.4f}")

    # Test avec différentes valeurs de pruning
    pruning_values = [1, 2, 3, 4, 5, 10, 15, 20]

    for pruning in pruning_values:
        # Création de la racine avec les données d'entraînement
        root = Node2(X_train, y_train)

        # Construction de l'arbre avec élagage
        root.grow(pruning=pruning)
        root.extend(pruning=pruning)

        # Calcul du taux de prédictions correctes sur les données de prédiction
        correct = 0
        total = len(X_pre)

        for i in range(total):
            prediction = root.predict(X_pre[i])
            if prediction == y_pre[i]:
                correct += 1

        print(f"Pruning = {pruning}, taux de prédiction : {correct/total:.4f}")

    # Commentaire sur les résultats
    print("Commentaire : Une valeur de pruning trop élevée fait chuter le taux de prédiction")
    print("car l'arbre devient trop simple et ne peut pas capturer toutes les nuances des données.")


# Exécution des fonctions
# Exercice 1
print("\n=== Exercice 1b ===")
set_data()
exercice1b()

print("\n=== Exercice 1c ===")
exercice1c()

print("\n=== Exercice 1d ===")
exercice1d()



# Exercice 2
print("\n=== Exercice 2a ===")
exercice2a()

print("\n=== Exercice 2b ===")
exercice2b()

print("\n=== Exercice 2c ===")
exercice2c()

print("\n=== Exercice 2d ===")
exercice2d()

print("\n=== Exercice 2e ===")
exercice2e()

print("\n=== Exercice 2f ===")
exercice2f()

print("\n=== Exercice 2g ===")
exercice2g()

print("\n=== Exercice 2h ===")
exercice2h()

print("\n=== Exercice 3 ===")
exercice3()
