from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import pandas as pd
import os, sys, math, csv, datetime, pickle, time, json


class UNetClassifier:
    def __init__(self, input_image_size, num_classes, save_logs):
        self.number_kernels = [64, 128, 256, 512, 1024]
        self.input_tensor = Input(shape=input_image_size)
        self.output_classes = num_classes
        self.save_history = save_logs

    def _create_cnn_block(self, input_tensor, num_filters, max_pool=True):
        output_tensor = Conv2D(filters=num_filters, kernel_size=(3,3), kernel_initializer='he_normal')(input_tensor)
        output_tensor = Conv2D(filters=num_filters, kernel_size=(3,3), kernel_initializer='he_normal')(output_tensor)
        if max_pool:
            output_tensor = MaxPooling2D(pool_size=(2,2))(output_tensor)

        return output_tensor

    def _add_dense(self, output_tensor):
        output_tensor = Flatten()(output_tensor)
        if self.output_classes == 2:
            output_tensor = Dense(1, activation='sigmoid')(output_tensor)
        else:
            output_tensor = Dense(self.output_classes, activation='softmax')

        return output_tensor

    def _create_model(self):
        output_tensor = self.input_tensor
        for index in range(len(self.number_kernels)):
            output_tensor = self._create_cnn_block(output_tensor, self.number_kernels[index], max_pool=index != (len(self.number_kernels) - 1))

        self.cnn_output = output_tensor
        self.final_output = self._add_dense(self.cnn_output)
        self.model = Model(inputs=[self.input_tensor], outputs=[self.final_output])

    def _compile_model(self):
        self.opt = Adam(lr=3e-5)
        if self.output_classes == 2:
            loss = 'log_loss'
        else:
            loss = 'categorical_crossentropy'
        self.model.compile(optimizer=self.opt, loss=loss, metrics=['acc'])

    def train(self, train_x, train_y, epochs, batch_size, validation_split):
        self.history = self.model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        if self.save_history:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            with open('logs/history.pkl', 'wb') as fin:
                pickle.dump(self.history, fin)
        return self.history

    def test(self, test_x, test_y):
        evaluation_results = self.model.evaluate(test_x, test_y)
        if self.save_history:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            with open('logs/test_evaluation', 'w') as fin:
                fin.write(str(evaluation_results))
        return evaluation_results

    def predict(self, test_x):
        prediction_classes = self.model.predict(test_x)
        if self.save_history:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            with open('logs/predictions', 'w') as fin:
                fin.write(str(prediction_classes))
        return prediction_classes
