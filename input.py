import os

import numpy as np
import scipy.io as sio
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf

DATA_DIR = 'data'


class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch >= self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


# Extra: 531131
# Train: 73257 - 14976 = 58240
# Total: extra+train = 589371
def get_data(num_training=58240, num_validation=14976, num_test=1000):
  X_train, X_test, y_train, y_test = _read_data()

  y_train[y_train > 9] = 0
  y_test[y_test > 9] = 0

  # Subsample the data
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]

  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]

  # mask = range(num_test)
  # X_test = X_test[mask]
  # y_test = y_test[mask]

  # perm = np.arange(num_training)
  # np.random.shuffle(perm)
  # X_train = X_train[perm]
  # y_train = y_train[perm]

  # Normalize the data: subtract the mean image
  # mean_image = np.mean(X_train, axis=0)
  # X_train -= mean_image
  # X_val -= mean_image
  # X_test -= mean_image

  train = DataSet(X_train, y_train)
  validation = DataSet(X_val, y_val)
  test = DataSet(X_test, y_test)

  return base.Datasets(train=train, validation=validation, test=test)


def _read_data():
  # xs, ys = _read_svhn_file(os.path.join(DATA_DIR, "extra_32x32.mat"))
  X_train, y_train = _read_svhn_file(os.path.join(DATA_DIR, "train_32x32.mat"))
  X_test, y_test = _read_svhn_file(os.path.join(DATA_DIR, "test_32x32.mat"))
  # X_train = np.concatenate(xs)
  # y_train = np.concatenate(ys)
  return X_train, X_test, y_train, y_test


def _read_svhn_file(filepath):
  print("Reading", filepath)
  datadict = sio.loadmat(filepath)
  y = datadict['y'].reshape(datadict['y'].shape[0],)
  return datadict['X'].transpose((3, 0, 1, 2)), y
