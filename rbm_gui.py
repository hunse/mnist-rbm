import os
import urllib

import numpy as np

import nengo
from nengo.dists import Choice

import find_neuron_params
import mnist

# --- parameters
presentation_time = 0.1
Ncode = 10
pstc = 0.006

# --- functions
def norm(x, **kwargs):
    return np.sqrt((x**2).sum(**kwargs))

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def forward(x, weights, biases):
    for w, b in zip(weights, biases):
        x = np.dot(x, w)
        x += b
        if w is not weights[-1]:
            x = sigmoid(x)

    return x

def get_image(t):
    return test_images[int(t / presentation_time) % len(test_images)]

def test_dots(t, dots):
    i = int(t / presentation_time) % len(test_images)
    j = np.argmax(dots)
    return test_labels[i] == vocab_labels[j]

# --- load the RBM data
rbm_file = 'rbm.npz'
if not os.path.exists(rbm_file):
    urllib.urlretrieve("http://files.figshare.com/1448053/rbm.npz", rbm_file)

rbm = np.load(rbm_file)
weights = rbm['weights']
biases = rbm['biases']

# --- load the testing data
[train_set, test_set] = mnist.load_data(train=True, valid=False, test=True)
train_images, train_labels = train_set
test_images, test_labels = test_set

# shuffle
rng = np.random.RandomState(92)
inds = rng.permutation(len(test_images))
test_images = test_images[inds]
test_labels = test_labels[inds]

# --- find average semantic pointers (codes) for each label
train_codes = forward(train_images, weights, biases)
vocab_labels = np.unique(train_labels)
vocab_codes = np.zeros((len(vocab_labels), train_codes.shape[-1]))
for i, label in enumerate(vocab_labels):
    vocab_codes[i] = train_codes[train_labels.flatten() == label].mean(0)

vocab_codes /= norm(vocab_codes, axis=1, keepdims=True)

# --- find good neuron parameters
neuron_params_file = 'neuron_params.npz'
if not os.path.exists(neuron_params_file):
    find_neuron_params.find_params(savefile=neuron_params_file)

# --- load and format neuron params
neuron_params = dict(np.load(neuron_params_file))
N = neuron_params.pop('N')
for p in ['encoders', 'max_rates', 'intercepts']:
    neuron_params[p] = Choice(neuron_params[p])
neuron_params['radius'] = neuron_params['radius'].item()

# --- create the model
model = nengo.Network()
with model:
    input_images = nengo.Node(output=get_image)

    # --- make sigmoidal layers
    layers = []
    output = input_images
    for w, b in zip(weights[:-1], biases[:-1]):
        layer = nengo.networks.EnsembleArray(N, b.size, **neuron_params)
        bias = nengo.Node(output=b)
        nengo.Connection(bias, layer.input, synapse=0)

        nengo.Connection(output, layer.input, transform=w.T, synapse=pstc)
        output = layer.add_output('sigmoid', function=sigmoid)

        layers.append(layer)

    # --- make code layer
    w, b = weights[-1], biases[-1]
    code = nengo.networks.EnsembleArray(Ncode, b.size)
    bias = nengo.Node(output=b)
    nengo.Connection(bias, code.input, synapse=0)
    nengo.Connection(output, code.input, transform=w.T, synapse=pstc)

    # --- make cleanup
    n_labels, n_codes = vocab_codes.shape
    dots = nengo.Node(output=lambda t, x: np.dot(vocab_codes, x),
                      size_in=n_codes, size_out=n_labels)
    nengo.Connection(code.output, dots, synapse=0.01)

    test = nengo.Node(output=test_dots, size_in=n_labels)
    nengo.Connection(dots, test)

    probe_dots = nengo.Probe(dots)
    probe_test = nengo.Probe(test)

if __name__ == '__main__':
    import nengo_gui
    nengo_gui.Viz(__file__).start()
