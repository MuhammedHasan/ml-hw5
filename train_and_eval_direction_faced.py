import os
# import pickle
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import theano

import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


class FaceDirectionClassifier(object):

    def __init__(self, mini_batch_size=10, epoch=60, pic_size=(30, 32),
                 learning_rate=0.01, activation_fn=network3.sigmoid,
                 filter_window_size=(5, 5), num_of_filter_window=20,
                 labels=('left', 'straight', 'right',  'up')):

        self.labels = labels
        self.directions = {j: i for i, j in enumerate(self.labels)}

        self.file_label_number = 1

        self.mini_batch_size = mini_batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn

        self._pic_size = pic_size
        self._num_of_filter_window = num_of_filter_window
        self._filter_window_size = filter_window_size

        self.net = Network([
            ConvPoolLayer(
                image_shape=(self.mini_batch_size, 1,
                             self._pic_size[0], self._pic_size[1]),
                filter_shape=(20, 1,
                              self._filter_window_size[0],
                              self._filter_window_size[1]),
                poolsize=(2, 2),
                activation_fn=self.activation_fn
            ),
            FullyConnectedLayer(
                n_in=self._conved_pic_size(),
                n_out=100,
                activation_fn=self.activation_fn
            ),
            SoftmaxLayer(
                n_in=100,
                n_out=4
            )
        ], self.mini_batch_size)

    def _conved_pic_size(self):
        pic_h = (self._pic_size[0] - self._filter_window_size[0] + 1) / 2
        pic_w = (self._pic_size[1] - self._filter_window_size[1] + 1) / 2
        return self._num_of_filter_window * pic_h * pic_w

    def _labels_to_number(self, label):
        return self.directions[label]

    def _number_to_label(self, number):
        return self.labels[number]

    def read_data(self):
        '''
        Read all data include train, test, and validation
        '''
        self.training_data = self._read_pics()
        self.validation_data = self._read_pics('ValidationSet')
        self.test_data = self._read_pics('TestSet')

    def _read_pics(self, folder="TrainingSet"):
        '''
        Reads all image and labels from given folder
        folder: TrainingSet or TestSet
        return: list of tuple of label and image matrix
        '''
        X, y = zip(*[
            (
                self._pic_to_features(np.asarray(
                    Image.open(folder + "/" + i).convert('L')
                ).astype(np.float64)),
                self._labels_to_number(i.split('_')[self.file_label_number])
            )
            for i in os.listdir(folder)
        ])
        X = theano.shared(np.asarray(X).astype(np.float64), name="x")
        y = theano.shared(np.asarray(y).astype(np.int32), name="y")
        return (X, y)

    def _pic_to_features(self, pic):
        '''
        Convert pictures to two time compressed feature array
        pic: np array of dimensions M x N
        return: np array of one dimensions with size M/4 * N/4
        '''
        return self.compress_image(self.compress_image(pic)).reshape(-1)

    def compress_image(self, pic):
        """
        Compress image to half size
        pic: np array of dimensions M x N
        return: np array of dimensions M/2 x N/2
        """
        return np.array([
            pic[m:m + 2, n:n + 2].sum() / 4.0
            for m in range(0, pic.shape[0], 2)
            for n in range(0, pic.shape[1], 2)
        ]).reshape((pic.shape[0] / 2, pic.shape[1] / 2))

    def predict(self, X):
        return [self.net.predict(x) for x in X]

    def train(self):
        evolation = self.net.SGD(
            self.training_data,
            self.epoch,
            self.mini_batch_size,
            self.learning_rate,
            self.validation_data,
            self.test_data)
        return list(evolation)

    def plot_evaluation(self, evaluation):
        ev = zip(*evaluation)
        plt.ylim([0, 1.1])
        if self.activation_fn == network3.sigmoid:
            plt.title('sigmoid')
        if self.activation_fn == network3.ReLU:
            plt.title('ReLU')
        plt.plot(ev[0], label='validation')
        plt.plot(ev[1], label='test')
        plt.legend(loc='lower right')
        return


if __name__ == '__main__':
    fc = FaceDirectionClassifier(activation_fn=network3.ReLU)
    fc.read_data()
    evolation = fc.train()

    print(evolation)
    print(len(evolation))

    plt.figure(1)
    fc.plot_evaluation(evolation)

    fc = FaceDirectionClassifier()
    fc.read_data()
    evolation = fc.train()
    plt.figure(2)
    fc.plot_evaluation(evolation)

    plt.show()

    print('-' * 25)
    print('Sigmoid is better then ReLU')
    print('''Because learning rate as 0.01 is too high for ReLU activation.
    Weigth are extramly updated in forward and backover operation
    It is too hard to converage with this learing rate
    but if we decrease learing rate to 0.0001
    performance of to ReLU will be same with sigmoid''')
