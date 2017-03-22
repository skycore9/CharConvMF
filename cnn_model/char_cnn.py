#!/usr/bin/env python
# -*- coding:utf-8 -*-

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Embedding
from keras.layers import ThresholdedReLU
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from data_utils import Data


class CNN_Module(object):
    def __init__(self, data_path, is_polarity=False):
        # CNN architecture
        self.conv_layers = [[256, 7, 3],
                            [256, 7, 3],
                            [256, 3, None],
                            [256, 3, None],
                            [256, 3, None],
                            [256, 3, 3]]
        self.fully_layers = [1024, 1024]
        self.l0 = 1014
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
        self.alphabet_size = len(self.alphabet)
        self.embedding_size = 70
        self.is_polarity = is_polarity
        self.th = 1e-6
        self.dropout_p = 0.5
        self.data_path = data_path
        if self.is_polarity:
            self.no_of_classes = 2
        else:
            self.no_of_classes = 5

        self._build_model()

    def _build_model(self):
        print "Building the model..."

        # building the model
        # Input layer
        inputs = Input(shape=(self.l0,), name='sent_input', dtype='int64')

        # Embedding layer
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.l0)(inputs)

        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = ThresholdedReLU(self.th)(x)
            if not cl[2] is None:
                x = MaxPooling1D(cl[2])(x)

        x = Flatten()(x)

        # Fully connected layers
        for idx, fl in enumerate(self.fully_layers):
            x = Dense(fl, name="output" + str(idx))(x)
            x = ThresholdedReLU(self.th)(x)
            x = Dropout(self.dropout_p)(x)

        predictions = Dense(self.no_of_classes, activation='softmax')(x)
        self.model = Model(input=inputs, output=predictions)
        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        print "Built"

    def train(self):
        print "Loading the data sets...",
        train_data = Data(data_source=self.data_path,
                          alphabet=self.alphabet,
                          no_of_classes=self.no_of_classes,
                          l0=self.l0,
                          is_polarity=self.is_polarity)

        train_data.loadData()
        X_train, y_train = train_data.getAllData()
        print "Loadded"

        print "Training ..."
        self.model.fit(X_train, y_train, nb_epoch=5000, batch_size=128, validation_split=0.2, callbacks=EarlyStopping)

        model_name = "cnn_model5.h5"
        if self.is_polarity:
            model_name = "cnn_model2.h5"
        self.model.save(model_name)

        print "Done!."
