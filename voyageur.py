import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from math import sqrt
import time

def gen_coord(N):
    """
    Génère les coordonnées (x, y) de N villes et un itinéraire initial.
    :param N: nombre de villes
    :return: (villes, itinéraire)
    """
    villes = [(np.random.rand(), np.random.rand()) for _ in range(N)]
    itineraire = list(range(N))
    return villes, itineraire

def gen_lignes(villes, itineraire):
    """
    Construit une liste de segments reliant les villes selon l'itinéraire.
    :param villes: liste des coordonnées des villes
    :param itineraire: liste de l'ordre de passage des villes
    :return: liste de segments (chaque segment est une liste de 2 tuples)
    """
    lignes = []
    for i in range(len(itineraire) - 1):
        a = villes[itineraire[i]]
        b = villes[itineraire[i+1]]
        lignes.append([a, b])
    return lignes

def longueur(lignes):
    """
    Calcule la longueur totale du parcours.
    :param lignes: liste de segments (chaque segment est une liste de 2 tuples)
    :return: somme des distances cartésiennes
    """
    total = 0
    for seg in lignes:
        (x1, y1), (x2, y2) = seg
        total += sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)
    return total

def trace_itineraire(villes, itineraire, titre, nom_fichier):
    """
    Trace l'itinéraire des villes et sauvegarde l'image.
    :param villes: liste des coordonnées des villes
    :param itineraire: liste de l'ordre de passage des villes
    :param titre: titre du graphique
    :param nom_fichier: nom du fichier image
    """
    lignes = gen_lignes(villes, itineraire)
    lc = LineCollection(lignes, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.scatter(*zip(*villes), c='red')
    ax.set_title(titre)
    plt.savefig(f"{nom_fichier}.png")
    plt.show()
    plt.close()

def trouve_ppv(villes, index, itineraireppv):
    """
    Trouve l'index de la ville la plus proche de la ville de référence,
    parmi celles qui ne sont pas encore dans itineraireppv.

    :param villes: liste des coordonnées des villes
    :param index: index de la ville de référence
    :param itineraireppv: liste des villes déjà visitées
    :return: index de la ville la plus proche
    """
    ville_ref = villes[index]
    min_dist = float('inf')
    ville_ppv = None

    for i in range(len(villes)):
        # On ne considère que les villes qui ne sont pas déjà dans l'itinéraire
        if i not in itineraireppv:
            ville = villes[i]
            # Calcul de la distance cartésienne
            distance = sqrt((ville[0] - ville_ref[0])**2 + (ville[1] - ville_ref[1])**2)

            if distance < min_dist:
                min_dist = distance
                ville_ppv = i

    return ville_ppv

def faire_ppv(villes):
    """
    Construit l'itinéraire selon l'algorithme du plus proche voisin.

    :param villes: liste des coordonnées des villes
    :return: itineraireppv - liste des indices des villes dans l'ordre du plus proche voisin
    """
    itineraireppv = [0]  # On commence par la première ville (indice 0)
    ville_ref = 0  # Ville de référence initiale

    # Tant qu'on n'a pas visité toutes les villes
    while len(itineraireppv) < len(villes):
        # Trouver la ville la plus proche de la ville de référence
        prochain = trouve_ppv(villes, ville_ref, itineraireppv)
        # Ajouter cette ville à l'itinéraire
        itineraireppv.append(prochain)
        # Cette ville devient la nouvelle référence
        ville_ref = prochain

    return itineraireppv

def gen_transposition(villes, itineraire):
    """Effectue une permutation aléatoire de deux villes si elle améliore la distance.

    Args:
        villes (list[tuple]): Coordonnées des villes.
        itineraire (list[int]): Itinéraire courant.

    Returns:
        tuple: (distance minimale, itinéraire correspondant)
    """
    # Choix de deux positions distinctes au hasard dans l'itinéraire
    i, j = np.random.choice(len(itineraire), size=2, replace=False)

    # Calcul de la distance actuelle
    lignes_avant = gen_lignes(villes, itineraire)
    dist_avant = longueur(lignes_avant)

    # Création d'une copie et permutation des deux positions
    itineraire_perm = itineraire.copy()
    itineraire_perm[i], itineraire_perm[j] = itineraire_perm[j], itineraire_perm[i]

    # Calcul de la distance après permutation
    lignes_apres = gen_lignes(villes, itineraire_perm)
    dist_apres = longueur(lignes_apres)

    if dist_apres < dist_avant:
        return dist_apres, itineraire_perm
    else:
        return dist_avant, itineraire

def gen_recuit(villes, itineraire, temps):
    """Applique une permutation aléatoire avec acceptation de type recuit simulé.

    Args:
        villes (list[tuple]): Coordonnées des villes.
        itineraire (list[int]): Itinéraire courant.
        temps (int): Temps discret influençant la température.

    Returns:
        tuple: (distance retenue, itinéraire correspondant)
    """
    # Choix de deux positions distinctes
    i, j = np.random.choice(len(itineraire), size=2, replace=False)

    # Distance avant permutation
    lignes_avant = gen_lignes(villes, itineraire)
    dist_avant = longueur(lignes_avant)

    # Itinéraire après permutation
    itin_perm = itineraire.copy()
    itin_perm[i], itin_perm[j] = itin_perm[j], itin_perm[i]
    lignes_apres = gen_lignes(villes, itin_perm)
    dist_apres = longueur(lignes_apres)

    # Choix aléatoire pour l'acceptation suivant t(x)=1/(1+x)
    alea = np.random.random()
    temperature = 1.0 / (1 + temps)

    if dist_avant <= dist_apres:
        dmin, itin_min = dist_avant, itineraire
        dmax, itin_max = dist_apres, itin_perm
    else:
        dmin, itin_min = dist_apres, itin_perm
        dmax, itin_max = dist_avant, itineraire

    if alea < temperature:
        return dmax, itin_max
    else:
        return dmin, itin_min

def gen_recuit_sim(villes, itineraire, temps, itinmax):
    """Applique une permutation aléatoire avec acceptation de type recuit simulé amélioré.

    Args:
        villes (list[tuple]): Coordonnées des villes.
        itineraire (list[int]): Itinéraire courant.
        temps (int): Temps discret influençant la température.
        itinmax (int): Nombre maximum d'appels à la fonction gen_recuit_sim.

    Returns:
        list: itinéraire final après optimisation
    """
    global mindistance, minitineraire, minidx

    # Choix de deux positions distinctes au hasard
    i, j = np.random.choice(len(itineraire), size=2, replace=False)
    if i > j:
        i, j = j, i  # On s'assure que i < j

    # Calcul de la distance avant modification
    lignes_avant = gen_lignes(villes, itineraire)
    dist_avant = longueur(lignes_avant)

    # Création d'un nouvel itinéraire selon le type de mélange
    itin_perm = itineraire.copy()

    # Tirage aléatoire pour déterminer le type de mélange
    type_melange = np.random.random()  # Nombre entre 0 et 1

    if type_melange < 0.10:
        # Cas 1 (10%): Permutation simple de deux villes
        itin_perm[i], itin_perm[j] = itin_perm[j], itin_perm[i]
    elif type_melange < 0.55:
        # Cas 2 (45%): Inversion de l'ordre des villes entre i et j inclus
        itin_perm[i:j+1] = itin_perm[i:j+1][::-1]
    else:
        # Cas 3 (45%): Permutation aléatoire des villes entre i et j inclus
        segment = itin_perm[i:j+1].copy()
        np.random.shuffle(segment)
        itin_perm[i:j+1] = segment

    # Calcul de la distance après modification
    lignes_apres = gen_lignes(villes, itin_perm)
    dist_apres = longueur(lignes_apres)

    # Calcul de la température selon la nouvelle fonction t
    temperature = 1.0 / (1 + temps / (itinmax / 200))

    # Détermination de l'itinéraire final selon le critère modifié
    alea = np.random.random()
    diff_dist = dist_avant - dist_apres
    proba = np.exp(3.5 * diff_dist * temperature)

    if alea < proba:
        # On prend la distance maximale (qui pourrait être moins bonne)
        if dist_avant >= dist_apres:
            itineraire_final = itineraire
            distance_finale = dist_avant
        else:
            itineraire_final = itin_perm
            distance_finale = dist_apres
    else:
        # On prend la distance minimale (qui est meilleure)
        if dist_avant <= dist_apres:
            itineraire_final = itineraire
            distance_finale = dist_avant
        else:
            itineraire_final = itin_perm
            distance_finale = dist_apres

    # Mise à jour des variables globales avec gestion de la dernière amélioration
    if (temps - minidx) > (0.04 * itinmax):
        # Si aucune amélioration depuis longtemps, on revient à la dernière sauvegarde
        if len(minitineraire) > 0:
            itineraire_final = minitineraire.copy()
    else:
        # Si on trouve une meilleure solution, on met à jour les variables globales
        if distance_finale < mindistance:
            mindistance = distance_finale
            minitineraire = itineraire_final.copy()
            minidx = temps

    return itineraire_final

def exercice1(villes, itineraire):
    """
    Affiche la distance totale, trace l'itinéraire et affiche le temps de calcul.
    :param villes: liste des coordonnées des villes
    :param itineraire: liste de l'ordre de passage des villes
    """
    start = time.time()
    lignes = gen_lignes(villes, itineraire)
    dist = longueur(lignes)
    print(f"Distance totale de l'itinéraire : {dist:.4f}")
    trace_itineraire(villes, itineraire, "Itinéraire initial", "itineraire_initial")
    end = time.time()
    print(f"Temps de calcul : {(end - start) * 1000:.2f} millisecondes")

def exercice2(villes):
    """
    Applique l'algorithme du plus proche voisin, affiche la distance totale,
    trace l'itinéraire et affiche le temps de calcul.

    :param villes: liste des coordonnées des villes
    """
    start = time.time()
    itineraireppv = faire_ppv(villes)
    lignes = gen_lignes(villes, itineraireppv)
    dist = longueur(lignes)
    print(f"Distance totale avec l'algorithme du plus proche voisin : {dist:.4f}")
    trace_itineraire(villes, itineraireppv, "Itinéraire avec l'algorithme du plus proche voisin", "itineraire_ppv")
    end = time.time()
    print(f"Temps de calcul : {(end - start) * 1000:.2f} millisecondes")
    return itineraireppv

def exercice3(villes, itineraire):
    """Améliore l'itinéraire par transpositions aléatoires."""
    start = time.time()
    dist = None
    for _ in range(100 * len(itineraire)):
        dist, itineraire = gen_transposition(villes, itineraire)

    print(f"Distance totale après transpositions : {dist:.4f}")
    trace_itineraire(
        villes,
        itineraire,
        "Itinéraire après transpositions",
        "itineraire_transpositions",
    )
    end = time.time()
    print(f"Temps de calcul : {(end - start) * 1000:.2f} millisecondes")
    return itineraire

def exercice4(villes, itineraire):
    """Optimisation par l'algorithme du recuit simple."""

    start = time.time()
    dist = None
    for t in range(5000 * len(itineraire)):
        dist, itineraire = gen_recuit(villes, itineraire, t)

    print(f"Distance totale après recuit : {dist:.4f}")
    trace_itineraire(
        villes,
        itineraire,
        "Itinéraire après recuit",
        "itineraire_recuit",
    )
    end = time.time()
    print(f"Temps de calcul : {(end - start) * 1000:.2f} millisecondes")
    return itineraire

def exercice5(villes, itineraire):
    """Optimisation par l'algorithme du recuit standard avec les variables globales."""
    global mindistance, minitineraire, minidx

    # Initialisation des variables globales
    lignes_initiales = gen_lignes(villes, itineraire)
    mindistance = longueur(lignes_initiales)
    minitineraire = itineraire.copy()
    minidx = 0

    start = time.time()
    itinmax = 5000 * len(itineraire)

    for t in range(itinmax):
        itineraire = gen_recuit_sim(villes, itineraire, t, itinmax)

    # Calcul de la distance finale
    lignes = gen_lignes(villes, itineraire)
    dist = longueur(lignes)

    print(f"Distance totale après recuit avec variables globales : {dist:.4f}")
    trace_itineraire(
        villes,
        itineraire,
        "Itinéraire après recuit avec variables globales",
        "itineraire_recuit_globals",
    )
    end = time.time()
    print(f"Temps de calcul : {(end - start) * 1000:.2f} millisecondes")
    return itineraire


if __name__ == "__main__":
    N = 40  # Nombre de villes
    villes, itineraire = gen_coord(N)
    print(f"=== Résolution du problème du voyageur de commerce pour {N} villes ===")

    print("\n=== Exercice 1: Itinéraire initial ===")
    exercice1(villes, itineraire)

    print("\n=== Exercice 2: Algorithme du plus proche voisin ===")
    itineraire_ppv = exercice2(villes)

    print("\n=== Exercice 3: Amélioration par transpositions ===")
    itineraire_transpose = exercice3(villes, itineraire_ppv)

    print("\n=== Exercice 4: Optimisation par recuit simple ===")
    itineraire_recuit = exercice4(villes, itineraire_transpose)

    print("\n=== Exercice 5: Optimisation par recuit avec variables globales ===")
    itineraire_final = exercice5(villes, itineraire_recuit)
