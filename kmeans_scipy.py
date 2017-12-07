import numpy as np
from scipy.cluster.vq import kmeans


def mapper(key, value):
    # key: None
    # value: one line of input file

    yield "key", value   # this is how you yield a key, value pair


def measureDistance(center, vector):
    return np.sum(np.abs(center - vector))


def reducer(key, values):
    k = 200
    max_values = 15000

    # Choose the first mean µ1 uniformly at random from the set X and add it to the set M.
    centers = []
    z = np.random.choice(len(values))
    centers.append(values[z])

    # For each point x ∈ X, compute the squared distance D(x) between x and the nearest mean µ in M.
    distances = np.zeros(len(values[:max_values]))
    for v, vector in enumerate(values[:max_values]):
        distances[v] = measureDistance(centers[0], vector)
    distances = np.square(distances)
    distances_norm = distances / np.sum(distances)

    # Choose the next mean µ randomly from the set X, where the probability
    # of a point x ∈ X being chosen is proportional to D(x), and add µ to M.
    distances_all = np.full((k, len(values[:max_values])), np.inf)
    while len(centers) < k:
        z = np.random.choice(np.arange(0, len(values[:max_values])), p=distances_norm)
        # z = np.argmax(distances_norm)
        centers.append(values[z])

        c_int = len(centers)-1
        for v, vector in enumerate(values[:max_values]):
            distances_all[c_int][v] = measureDistance(values[z], vector)
        distances = np.amin(distances_all[:c_int], axis=0)  # get distance of nearest point
        distances = np.square(distances)
        distances_norm = distances / np.sum(distances)
    #     if len(centers) % 10 == 0:
    #         print('centers sampled: ' + str(len(centers)))
    # print('initialisation done')

    # Perform kmeans computation on this dataset
    centers_new, _ = kmeans(values, centers, iter=30)
    # centers_new, _ = kmeans2(values, centers, iter=10, minit='matrix')
    yield centers_new
