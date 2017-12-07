import numpy as np
np.random.seed(34)


k = 200  # 9 * 23 = ~200


def measureDistance(center, vector):
    # return np.linalg.norm(center - vector)
    return np.sum(np.abs(center - vector))


def mapper(key, value):
    # key: None
    # value: one line of input file
    #        2D numpy array, shape (3000, 250)
    yield 'key', value
    # yield 'key', vectors_by_centers


def reducer(key, values):
    # values: maprun -> center -> vectors
    global k
    vectors = values
    del values

    # Choose the first mean µ1 uniformly at random from the set X and add it to the set M.
    centers = []
    z = np.random.choice(len(vectors))
    centers.append(vectors[z])

    # For each point x ∈ X, compute the squared distance D(x) between x and the nearest mean µ in M.
    distances = np.zeros(len(vectors))
    for v, vector in enumerate(vectors):
        distances[v] = measureDistance(vector, centers[0])
    distances = np.square(distances)
    distances_norm = distances / np.sum(distances)

    # Choose the next mean µ randomly from the set X, where the probability
    # of a point x ∈ X being chosen is proportional to D(x), and add µ to M.
    while len(centers) < k:
        z = np.random.choice(np.arange(0, len(vectors)), p=distances_norm)
        # z = np.argmax(distances_norm)
        centers.append(vectors[z])

        distances_all = np.zeros((len(centers), len(vectors)))
        for c, center in enumerate(centers):
            for v, vector in enumerate(vectors):
                distances_all[c][v] = measureDistance(center, vector)
        distances = np.amin(distances_all, axis=0)  # get distance of nearest point
        distances = np.square(distances)
        distances_norm = distances / np.sum(distances)
    print('initialisation done')

    # Apply the standard k-means algorithm, initialized with these means.
    # nearest center/distance for [vector v]
    nearestCenters = [None] * vectors.shape[0]
    nearestDistances = [1e99] * vectors.shape[0]
    # center_sets = [[]] * k

    for v, vector in enumerate(vectors):
        if v % 100 == 0:
            print('on vector ' + str(v))
        for c, center in enumerate(centers):
            dist = measureDistance(center, vector)

            if (dist < nearestDistances[v] or nearestCenters[v] is None):
                nearestDistances[v] = dist
                nearestCenters[v] = c
    print('center sets done')

    nearestCenters = np.array(nearestCenters)
    # TODO this part can be directly inplemented below
    # for c, center in enumerate(centers):
    #     vectors_indices = np.where(nearestCenters == c)[0]
    #     if len(vectors_indices) > 0:
    #         center_sets[c] = np.take(vectors, vectors_indices, axis=0)
    # print('refactoring center sets done')

    # kmeans
    new_centers = []
    # for center_set in center_sets:
    for c, center in enumerate(centers):
        # if len(center_set) == 0:
            # continue
        vectors_indices = np.where(nearestCenters == c)[0]
        if len(vectors_indices) == 0:
            continue
        new_centers.append(
            np.mean(
                np.take(vectors, vectors_indices, axis=0),
                axis=0
            )
        )
        # new_center = np.mean(center_set, axis=0)
        # new_centers.append(new_center)

    print("Centers: {}".format(len(new_centers)))

    # merge nearest centers
    # while len(new_centers) > 200:
    #     distances = np.full((len(new_centers)), np.inf)
    #     # TODO can be optimized if not chosen by random, rather by smallest distances (but much slower)
    #     z = np.random.choice(len(new_centers))
    #     for j in range(len(new_centers)):
    #         if j == z:
    #             continue
    #         distances[j] = measureDistance(new_centers[z], new_centers[j])
    #
    #     ind = np.argmin(distances)
    #     new_center = np.mean([new_centers[z], new_centers[ind]], axis=0)
    #
    #     new_centers = np.delete(new_centers, [z, ind], axis=0)
    #     new_centers = np.append(new_centers, [new_center], axis=0)

    # Output: 200 vectors representing the selected centers
    #         each being 250 floats
    yield new_centers[:200]  # TODO k=200 should make this unnecessary
    # yield np.random.randn(200, 250)
