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
    print(f"Temps de calcul : {end - start:.4f} secondes")

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
    print(f"Temps de calcul : {end - start:.4f} secondes")
    return itineraireppv

if __name__ == "__main__":
    N = 40  # Nombre de villes
    villes, itineraire = gen_coord(N)
    print(f"=== Résolution du problème du voyageur de commerce pour {N} villes ===")

    print("\n=== Exercice 1: Itinéraire initial ===")
    exercice1(villes, itineraire)

    print("\n=== Exercice 2: Algorithme du plus proche voisin ===")
    itineraire_ppv = exercice2(villes)




