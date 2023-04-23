import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

def test(X_test, y_test, model, LE):
    num_pass = 0
    for i in range(len(X_test)):
        chunked_test_data = X_test[i]
        ground_truth_label = y_test[i]
        chunked_test_data = np.expand_dims(chunked_test_data, axis=0)
        # chunked_test_data = np.expand_dims(chunked_test_data, axis=-1)
        predicted_labels = np.argmax(model.predict(chunked_test_data), axis=-1)
        # print(predicted_labels)

        # likely_label = np.array(most_frequent(predicted_labels))
        # likely_label = np.expand_dims(predicted_labels, axis=-1)

        # prediction_class = np.squeeze(LE.inverse_transform(predicted_labels), axis=-1)
        # prediction_class = np.squeeze(predicted_labels, axis=-1)
        prediction_class = predicted_labels[0]
        # print('predicted class: ', prediction_class)
        if np.argmax(ground_truth_label) == prediction_class:
            print('PASS', end=' ')
            num_pass += 1
        print(f'y={np.argmax(ground_truth_label)}')
    accuracy = num_pass/len(X_test)
    print(f'accuracy = {accuracy}')

def create_model(num_filters=32, kernel_size=3, pool_size1=2, pool_size2=3, dropout_rate=0.25, input_shape=(96, 44, 1)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(num_filters, kernel_size, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size1, pool_size1), padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=num_filters*2, kernel_size=(kernel_size, kernel_size), activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size1, pool_size1), padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=num_filters*4, kernel_size=(kernel_size, kernel_size), activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size2, pool_size2), padding='same'))
    model.add(tf.keras.layers.Conv2D(filters=num_filters*8, kernel_size=(kernel_size, kernel_size), activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size2, pool_size2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'val_accuracy'])
    return model

def val_accuracy_scorer(model, X, y):
    _, val_acc = model.evaluate(X, y, verbose=0)
    return val_acc

def grid_search(X_train, y_train, X_test, y_test, param_grid):
    actual_model = create_model
    model = KerasClassifier(build_fn=actual_model, epochs=10, batch_size=32, verbose=0)

    # Create a GridSearchCV object
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

    # Fit the grid search object to the training data
    grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test))

    # Print the best parameters and accuracy score
    print("Best parameters: ", grid_result.best_params_)
    print("Best accuracy score: ", grid_result.best_score_)