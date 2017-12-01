import numpy as np
np.random.seed(34)


def measureDistance(center, vector):
    sm = 0
    for i, v in enumerate(vector):
        sm += abs(center[i] - v)
    return sm


def mapper(key, value):
    # key: None
    # value: one line of input file
    #        2D numpy array, shape (3000, 250)
    k = 23

    # initiate random centers (uniform distribution)
    centers = np.random.randn(k, 250)

    # nearest center/distance for [vector v]
    nearestCenters = [None] * value.shape[0]
    nearestDistances = [1e99] * value.shape[0]
    vectors_by_centers = [[]] * k

    for v, vector in enumerate(value[:3000]):
        for c, center in enumerate(centers):
            dist = measureDistance(center, vector)

            if (dist < nearestDistances[v] or nearestCenters[v] is None):
                nearestDistances[v] = dist
                nearestCenters[v] = c

    nearestCenters = np.array(nearestCenters)
    for c, center in enumerate(centers):
        vectors_indices = np.where(nearestCenters == c)[0]
        # print(vectors_indices)
        # print("center {} has {} 'nearest' vectors".format(c, len(vectors_indices)))
        if len(vectors_indices) > 0:
            vectors_by_centers[c] = np.take(value, vectors_indices, axis=0)

    yield 'key', vectors_by_centers


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    # values: maprun -> center -> vectors
    k = 23
    new_centers = np.zeros((k, 250))
    # print(len(values))  # 9
    # print(len(values[0]))  # 30
    # print(len(values[0][0]))  # diff

    new_centers = np.zeros((9 * k, 250))
    i = 0
    for center_sets in values:
        for center_set in center_sets:
            if len(center_set) == 0:
                continue

            new_center = np.mean(center_set, axis=0)
            new_centers[i] = new_center
            i += 1

    # print(new_centers.shape)
    # print(new_centers[:200, :].shape)

    # TODO instead of crop, merge nearest centers

    # Output: 200 vectors representing the selected centers
    #         each being 250 floats
    yield new_centers[:200, :]
    # yield np.random.randn(200, 250)
