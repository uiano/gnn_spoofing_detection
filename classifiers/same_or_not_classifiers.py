import os

import matplotlib.pyplot as plt
import numpy as np

import gsim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import keras.api._v2.keras as keras
import tensorflow as tf
from keras.api._v2.keras import Model
from keras.api._v2.keras.layers import Dense, Flatten
from tqdm.keras import TqdmCallback


class SameOrNotClassifier():

    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    @staticmethod
    def build_dataset(t_power, num_entries):
        # It returns a tensor t_dataset of shape (num_entries, 2, num_feat) and
        # a vector v_same of length num_entries. Each entry of t_dataset is a
        # matrix of the form:
        #
        # [ t_power[ind_tx_1, :, ind_sample_1]; t_power[ind_tx_2, :,
        # ind_sample_2 ]
        #
        # if ind_tx_1 == ind_tx_2, then the corresponding entry of v_same is
        # True, else it is False.
        #
        # v_same contains num_entries/2 entries = True and num_entries/2 entries
        # = False
        #

        def draw_no_replacement(high):
            # Returns two different uniformly distributed random integers
            # between 0 and high-1.
            ind_1 = gsim.rs.randint(high)
            ind_2 = gsim.rs.randint(high - 1)
            if ind_2 >= ind_1:
                ind_2 += 1
            return ind_1, ind_2

        num_entries_each_class = int(np.floor(num_entries / 2))

        num_tx, _, num_samples = t_power.shape
        lv_feat_pairs = []
        l_same = []
        for ind_entry in range(num_entries):

            if ind_entry < num_entries_each_class:
                ind_tx_1, ind_tx_2 = draw_no_replacement(num_tx)
                same = False
            else:
                ind_tx_1 = gsim.rs.randint(num_tx)
                ind_tx_2 = ind_tx_1
                same = True
            ind_sample_1, ind_sample_2 = gsim.rs.randint(num_samples,
                                                         size=(2, ))
            t_feat_pairs = np.stack(
                (t_power[ind_tx_1, :, ind_sample_1], t_power[ind_tx_2, :,
                                                             ind_sample_2]))
            lv_feat_pairs.append(t_feat_pairs)
            l_same.append(same)

        # Shuffle
        t_feat_pairs = np.array(lv_feat_pairs)
        v_same = np.array(l_same)
        v_ind = gsim.rs.permutation(len(lv_feat_pairs))
        return t_feat_pairs[v_ind], v_same[v_ind]

    def train(self, t_power, num_pairs_train, num_pairs_val, val_split=0.2):
        """ num_pairs_train training examples are constructed from a fraction
        (1-validation_split) of the outer slices of t_power. Conversely, num_pairs_val
        validation examples are constructed from the remaining fraction
        validation_split of the outer slices of t_power.
        
        """

        num_tx_val = int(np.floor(val_split * len(t_power)))
        if self.verbosity >= 2:
            print(
                f'Using {len(t_power)-num_tx_val} tx. positions for actual training and {num_tx_val} for validation.'
            )
        if num_tx_val == 0:
            raise ValueError
        v_ind = gsim.rs.permutation(len(t_power))
        t_power_train = t_power[v_ind[num_tx_val:]]
        t_power_val = t_power[v_ind[:num_tx_val]]

        t_feat_pairs_train, v_same_train = self.build_dataset(
            t_power_train, num_pairs_train)
        t_feat_pairs_val, v_same_val = self.build_dataset(
            t_power_val, num_pairs_val)

        self._train(t_feat_pairs_train=t_feat_pairs_train,
                    v_same_train=v_same_train,
                    t_feat_pairs_val=t_feat_pairs_val,
                    v_same_val=v_same_val)


class DnnSameOrNotClassifier(SameOrNotClassifier):

    class SymmetricNet(Model):

        learning_rate = 1e-4
        activation = 'leaky_relu'
        neurons_per_layer = 512

        def __init__(self):
            super().__init__()
            self.flatten = Flatten()
            self.d1 = Dense(self.neurons_per_layer,
                            activation=self.activation,
                            kernel_regularizer=tf.keras.regularizers.L1(.14))
            self.d2 = Dense(self.neurons_per_layer, activation=self.activation)
            self.d3 = Dense(self.neurons_per_layer, activation=self.activation)
            self.dout = Dense(1)

        def call(self, x):
            x_reversed = tf.reverse(x, [1])
            return (self.asymmetric_call(x) +
                    self.asymmetric_call(x_reversed)) / 2

        def asymmetric_call(self, x):
            # v_ means batch of vectors

            v_feat_2 = x[:, 0, :] - x[:, 1, :]

            v_feat_3 = tf.reshape(x, (-1, x.shape[1] * x.shape[2]))

            x = tf.concat((v_feat_2, v_feat_3), axis=1)  #

            x = self.d1(x)
            x = self.d2(x)
            x = self.d3(x)
            return self.dout(x)

    class CallbackStop(tf.keras.callbacks.Callback):
        min_accuracy = 0.98

        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > self.min_accuracy):
                self.model.stop_training = True

    def __init__(self, num_epochs=100, run_eagerly=False, **kwargs):

        super().__init__(**kwargs)
        self.model = self.SymmetricNet()
        self.num_epochs = num_epochs
        self.run_eagerly = run_eagerly

    def __str__(self):
        return "DNNC"

    def _train(self, t_feat_pairs_train, v_same_train, t_feat_pairs_val,
               v_same_val):

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model.learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=keras.metrics.BinaryAccuracy(name='accuracy',
                                                 threshold=0.))

        self.model.run_eagerly = self.run_eagerly
        history = self.model.fit(
            t_feat_pairs_train,
            v_same_train.astype(int),
            epochs=self.num_epochs,
            verbose=0,
            validation_data=(t_feat_pairs_val, v_same_val.astype(int)),
            batch_size=32,
            callbacks=[
                TqdmCallback(verbose=0),
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=10,
                                                 restore_best_weights=True,
                                                 mode='max'),
                # DnnSameOrNotClassifier.CallbackStop()
            ])

        def plot_loss(history):
            plt.subplot(211)
            plt.plot(history.history['loss'], '--', label='loss')
            plt.plot(history.history['val_loss'], '-', label='val_loss')
            #plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            print(
                f"{len(t_feat_pairs_train)} training examples, {len(t_feat_pairs_val)} validation examples"
            )
            plt.grid(True)

            plt.subplot(212)
            plt.plot(history.history['accuracy'], '--', label='accuracy')
            plt.plot(history.history['val_accuracy'],
                     '-',
                     label='val_accuracy')
            plt.ylim([0, 1.1])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)

        if self.verbosity:
            plot_loss(history)

            plt.figure()
            v_weights = np.sort(
                np.abs(self.model.d1.weights[0].numpy().ravel()))
            plt.hist(v_weights, bins=40)
            plt.title("Histogram of the weights of the first layer")
            print("abs(weights 1st layer) = ", v_weights)
            return

    def are_the_same(self, t_feat_pairs):
        # t_feat_pairs is num_pairs x 2 x num_feat returns a vector v_same of
        # length num_pairs where v_same[i] is True if t_feat_pairs[i][0] and
        # t_feat_pairs[i][1] are determined to belong to the same transmitter
        return np.ravel(self.model(t_feat_pairs).numpy() > 0)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path, num_feat):
        self.are_the_same(np.zeros((1, 2, num_feat)))
        self.model.load_weights(path)
