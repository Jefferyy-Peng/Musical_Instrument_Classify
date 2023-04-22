import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
import librosa
import librosa.display
import pickle
import fileinput
# import IPython.display as ipd

import tensorflow as tf
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