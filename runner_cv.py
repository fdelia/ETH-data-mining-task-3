# MIT License
#
# Copyright (c) 2016 Olivier Bachem
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Map Reduce Framework for Python (Task2, DM2016)

https://project.las.ethz.ch/task2

...

CV Version
"""
from collections import defaultdict
from itertools import chain, islice, izip
import argparse
import glob
import imp
import multiprocessing
import numpy as np
import os
import random
import sys
import logging
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
from io import BytesIO


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield list(chain([first], islice(iterator, size - 1)))


def isolated_batch_call(f, arguments):
    """Calls the function f in a separate process and returns list of results"""

    def lf(q):
        ret = f(*arguments)
        q.put(list(ret))

    ret = f(*arguments) # for debugging

    # q = multiprocessing.Queue()
    # p = multiprocessing.Process(target=lf, args=(q, ))
    # p.start()
    # ret = q.get()
    # p.join()
    return ret


def mapreduce(input, mapper, reducer, batch_size=50, log=False):
    """Python function that runs a worst-case map reduce framework on the provided data

    Args:
      input -- list or generator of (key, value) tuple
      mapper -- function that takes (key, value) pair as input and returns iterable key-value pairs
      reducer -- function that takes key + list of values and outputs (key, value) pair
      log -- whether log messages should be generated (default: False)

    Returns list of (key, value pairs)
    """
    # Set initial random seed
    random.seed(0)
    # Run mappers
    if log: logger.info("Starting mapping phase!")

    d = defaultdict(list)
    for pairs_generator in chunks(input, batch_size):
        pairs = np.array(pairs_generator)
        if log: logger.debug("  Running mapper for '%s' key with value '%s'...", k, v)
        for k2, v2 in isolated_batch_call(mapper, (None, pairs)):
        # for k2, v2 in mapper(None, pairs):
            if log: logger.debug("    Mapper produced (%s, %s) pair...", k2, v2)
            if not isinstance(k2, (basestring, int, float)):
                raise Exception("Keys must be strings, ints or floats (provided '%s')!"% k2)
            d[k2].append(v2)
    if log: logger.info("Finished mapping phase!")

    # Random permutations of both keys and values.
    keys = d.keys()
    random.shuffle(keys)
    for k in keys:
        random.shuffle(d[k])
    # Run reducers
    if log: logger.info("Starting reducing phase!")
    res = []
    if len(keys) > 1:
        raise Exception("Only one distinct key expected from mappers.")
    k = keys[0]
    v = np.vstack(d[k])

    r = isolated_batch_call(reducer, (k, v))
    if log: logger.debug("    Reducer produced %s", r)
    logger.info("Finished reducing phase!")
    return r


def yield_pattern(path):
    """Yield lines from each file in specified folder"""
    for i in glob.iglob(path):
        if os.path.isfile(i):
            with open(i, "r") as fin:
                for line in fin:
                    yield None, line


def import_from_file(f):
    """Import code from the specified file"""
    mod = imp.new_module("mod")
    exec f in mod.__dict__
    return mod


def evaluate(points, centers):
    score = 0.0
    for chunk in chunks(points, 20):
        batch = np.array(chunk)
        score += np.square(batch[:,np.newaxis,:] - centers).sum(axis=2).min(axis=1).sum()
    return score / points.shape[0]


def run(sourcestring, training_file, test_file, K, batch, log):
    mod = import_from_file(sourcestring)
    data = np.load(training_file)
    if test_file != '':
        t = np.load(test_file)
        data = np.concatenate((data, t), axis=0)
    print("Number of rows: {}".format(len(data)))

    # split input K times into sets K-1/K and 1/K parts
    scores = []
    for k in range(K):
        from_n = int(len(data) * k/K)
        to_n = int(len(data) * (k+1)/K)

        test_data = data[from_n : to_n]
        train_data = np.concatenate((data[0 : from_n], data[to_n : len(data)]))
        # print(len(train_data))
        # print(len(test_data))

        # Train
        output = mapreduce(train_data, mod.mapper, mod.reducer, batch, log)
        centers = np.vstack(output)

        # Score
        # score = evaluate(weights, test_data, mod.transform)
        score = evaluate(test_data, centers)
        scores.append(score)
        print("K {} of {} done, score: {}".format(k+1, K, score))

    print("mean score: {}".format(np.mean(scores)))

    # output = mapreduce(input, mod.mapper, mod.reducer, batch, log)
    # weights = np.array(output)
    # if weights.shape[0] > 1:
    #     logging.error("Incorrect format from reducer")
    #     sys.exit(-2)
    # test_data = np.loadtxt(test_file, delimiter=" ")
    #return evaluate(weights, test_data, mod.transform)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('train_file', help='File with the training instances')
    parser.add_argument('test_file', help='File with the testing instances')
    parser.add_argument('-K', help='Number of splits', default=3)
    parser.add_argument(
        'source_file', help='.py file with mapper and reducer function')
    parser.add_argument(
        '--log',
        '-l',
        help='Enable logging for debugging',
        action='store_true')
    args = parser.parse_args()
    BATCH = 2000

    try:
        K = int(args.K)
    except ValueError:
        print('Argument K is not an integer. See --help')
        sys.exit(-1)

    with open(args.source_file, "r") as fin:
        source = fin.read()

    run(source, args.train_file, args.test_file, K, BATCH, args.log)


if __name__ == "__main__":
    main()
