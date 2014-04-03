
import gzip
import os
import urllib

import numpy as np

# train_files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]
# test_files = ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

# def download_mnist_data(train=None, test=None):
#     base = "http://yann.lecun.com/exdb/mnist/"

#     if train is None:
#         train = not all(os.path.exists(f) for f in train_files)
#     if test is None:
#         test = not all(os.path.exists(f) for f in test_files)

#     files = (train_files if train else []) + (test_files if test else [])
#     for filename in files:
#         url = base + filename
#         urllib.urlretrieve(url, filename)

filename = "mnist.pkl.gz"


def download_data(force=False):
    base = "http://deeplearning.net/data/mnist/"
    if not os.path.exists(filename) or force:
        url = base + filename
        urllib.urlretrieve(url, filename)


def load_data(train=True, valid=False, test=True):
    import cPickle as pickle
    download_data()

    with gzip.open(filename, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    return (([train_set] if train else []) +
            ([valid_set] if valid else []) +
            ([test_set] if test else []))
