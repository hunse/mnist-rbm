mnist-rbm
=========

A spiking-neuron version of an RBM trained on the MNIST dataset


Requirements
------------

This model requires Nengo to be installed. Additionally, it requires Numpy
and Matplotlib, but since these are also requirements of Nengo they should
already be installed.


Running
-------

To run the model, simply execute `rbm.py`. It will do the following tasks:

  1. Define configurable model parameters.
  2. Define general helper functions
  3. Download and load the trained RBM weights and biases from Figshare.
  4. Download and load the MNIST testing data.
  5. Find the average semantic pointers for each label.
  6. Find good neuron parameters for representing a sigmoid.
  7. Create the Nengo model.
  8. Run the Nengo model and cache the result to `rundata.npz`, or load the
     cached result if it exists.
  9. Plot the results.

Please be patient when first running the script, as running the model can
take a while (but probably not more than ~10 minutes). Rerunning the script
should be fast, since it will use the cached data. Deleting the cached data
will cause the model to rerun entirely.
