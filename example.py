import numpy as np


def mapper(key, value):
    # key: None
    # value: one line of input file
    # value: 2D numpy array

    yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    # Output: 200 vectors representing the selected centers
    #   each being 250 floats

    yield np.random.randn(200, 250)
