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
from Data_preprocess import preprocess_train_data, preprocess_test_data
from testing import test, grid_search


def calc_fft(y, sr):
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1 / sr)
    Y = abs(np.fft.rfft(y) / n)
    return (Y, freq)


def Envelope(y, rate, threshold):
    mask = []
    # we want a rolling window so we create series as it is easy with it
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 4), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=1, ncols=4, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(4):
            axes[x].set_title(list(signals.keys())[i])
            axes[x].plot(list(signals.values())[i])
            axes[x].get_xaxis().set_visible(False)
            axes[x].get_yaxis().set_visible(False)
            i += 1

    plot_signals(signals)
    plt.show()

# def MFCC_Extractor(audio, sample_rate):
#     # for CNN
#     # print(label)
#     # fig,axs = plt.subplots(nrows=2, ncols=1)
#     # librosa.display.waveshow(audio_raw, sr=sample_rate, ax=axs[0])
#     # librosa.display.waveshow(audio, sr=sample_rate, ax=axs[1])
#     # plt.show()
#     mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=96)
#     # mfcc_dif1 = librosa.feature.delta(mfcc)
#     # mfcc_dif2 = librosa.feature.delta(mfcc_dif1)
#     # mfcc_d1_d2 = np.concatenate([mfcc,mfcc_dif1,mfcc_dif2],axis=0)
#     return mfcc

# def chunk_data(data, segment_length=50, overlap=0.5):
#     # Calculate the number of samples per segment and the number of overlapping samples
#     overlap_samples = int(overlap * segment_length)
#
#     # Calculate the total number of segments
#     total_segments = int(np.ceil(data.shape[0]/(segment_length - overlap_samples)))
#     # Initialize an empty list to store the segments
#     chunked_data = []
#     for i in range(total_segments):
#         # Calculate the start and end samples of the current segment
#         start = int(i * (segment_length - overlap_samples))
#         end = int(start + segment_length)
#
#         # Extract the current segment from the audio data
#         segment = data[start:end]
#
#         # Append the segment to the list of segments
#         if segment.shape[0] == segment_length:
#             chunked_data.append(segment)
#     return chunked_data

def most_frequent(arr):
    unique, counts = np.unique(arr, return_counts=True)
    most_freq_index = np.argmax(counts)
    return unique[most_freq_index]

def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "accuracy")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_accuracy")
    plt.title("train_accuracy vs val_accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()

np.random.seed(2023)
tf.random.set_seed(2023)

train_df = pd.read_csv("/Users/amadeus/Downloads/archive/Metadata_Train.csv")
test_df = pd.read_csv("/Users/amadeus/Downloads/archive/Metadata_Test.csv")

is_process_data = 1
is_train_model = 1
_is_test_model = 1
_is_grid_search = 0
_is_split = False
_is_trim = False
is_print_class_num = True
model_name = 'CNN_2C2F_8C_64F_44pool_dropout_before_mp_Ver'
header_name = '2C2F_8C_64F_44pool_dropout_before_mp_model.h'

if __name__ == '__main__':
    if is_process_data:
        Train_data_frame = preprocess_train_data(train_df, _is_split=_is_split, _is_trim=_is_trim, verbose=True)
        Test_data_frame = preprocess_test_data(test_df, _is_split=_is_split, _is_trim=_is_trim, verbose=True)
        pickle.dump(Train_data_frame, open(os.path.join('./music_classification', f'no_spilt_no_trim_no_overlap_chunk_MFCC.p'), 'wb'))
        pickle.dump(Test_data_frame, open(os.path.join('./music_classification', f'no_spilt_no_trim_no_overlap_chunk_MFCC_test.p'), 'wb'))
    else:
        Train_data_frame = pickle.load(open(os.path.join('./music_classification', f'no_spilt_no_trim_no_overlap_chunk_MFCC.p'), 'rb'))
        Test_data_frame = pickle.load(open(os.path.join('./music_classification', f'no_spilt_no_trim_no_overlap_chunk_MFCC_test.p'), 'rb'))
    sample_rate = 22050

    if is_print_class_num:
        num_drum = 0
        num_guitar = 0
        num_piano = 0
        num_violin = 0
        for index_num,row in tqdm(Train_data_frame.iterrows()):
            final_class_label = row['class']
            if final_class_label == 'Sound_Drum':
                num_drum += 1
            elif final_class_label == 'Sound_Guitar':
                num_guitar += 1
            elif final_class_label == 'Sound_Piano':
                num_piano += 1
            else:
                num_violin += 1
        print("num_drum = " + str(num_drum))
        print("num_guitar = " + str(num_guitar))
        print("num_piano = " + str(num_piano))
        print("num_violin = " + str(num_violin))

    X_train=np.array(Train_data_frame['features'].tolist())
    X_test=np.array(Test_data_frame['features'].tolist())
    y_train=np.array(Train_data_frame["class"].tolist())
    y_test=np.array(Test_data_frame["class"].tolist())
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    LE=LabelEncoder()
    y_train=to_categorical(LE.fit_transform(y_train))
    y_test=to_categorical(LE.fit_transform(y_test))

    num_channels = 1
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.02,random_state=42)
    X_train = np.expand_dims(X_train, axis = -1)
    X_test = np.expand_dims(X_test, axis = -1)
    print("Shape Of X_train:",X_train.shape)
    print("Shape Of X_test:",X_test.shape)
    print("Shape Of y_train:",y_train.shape)
    print("Shape Of y_test:",y_test.shape)
    num_labels=y_train.shape[1]

    num_classes = 4
    if is_train_model:
        model = tf.keras.models.Sequential([
            # First Conv2D layer with 64 filters and a 3x3 kernel size
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
            # First max pooling layer with 2x2 pool size
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling2D(pool_size=(4,4), padding='same'),
            #first dropout
            # Second Conv2D layer with 128 filters and a 3x3 kernel size
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
            # Second max pooling layer with 2x2 pool size
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.MaxPooling2D(pool_size=(4,4), padding='same'),
            # second dropout
            # Third Conv2D layer with 256 filters and a 3x3 kernel size
            # tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
            # # Third max pooling layer with 3x3 pool size
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.MaxPooling2D(pool_size=(4,4), padding='same'),
            # 3rd dropout
            # Fourth Conv2D layer with 640 filters and a 3x3 kernel size
            # tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
            # # Fourth max pooling layer with 3x3 pool size
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.MaxPooling2D(pool_size=(3,3), padding='same'),
            # Flatten the output of the previous layer
            tf.keras.layers.Flatten(),
            # Fully connected layer with 128 hidden units
            tf.keras.layers.Dense(units=64, activation='relu'),
            # Output layer with softmax activation function
            tf.keras.layers.Dense(units=num_classes, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpointer=ModelCheckpoint(filepath=os.path.join('./music_classification/models', f'{model_name}.hdf5'), monitor='val_accuracy', mode='max', save_best_only=True)
        start=datetime.now()
        history = model.fit(X_train, y_train, batch_size=32, epochs=40, validation_data=(X_test, y_test),callbacks=checkpointer)

        duration=datetime.now()-start
        print("Training Completed in time: ",duration)

        plot_accuracy_loss(history)
    model = tf.keras.models.load_model(f'./music_classification/models/{model_name}.hdf5')
    # Convert the model to the TensorFlow Lite format without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    if _is_grid_search:
        param_grid = {'num_filters': [32, 64],
                      'kernel_size': [3, 5],
                      'pool_size1': [2, 3],
                      'pool_size2': [3, 4],
                      'dropout_rate': [0.25, 0.5],
                      'input_shape': [(X_train.shape[1], X_train.shape[2], 1)]}

        grid_search(X_train, y_train, X_test, y_test, param_grid)




    # Save the model to disk
    open(f"./music_classification/models/{model_name}.tflite", "wb").write(tflite_model)

    basic_model_size = os.path.getsize(f"./music_classification/models/{model_name}.tflite")
    print("Model is %d bytes" % basic_model_size)
    a = 'echo "'
    b = "const unsigned char model[] = { "
    c = '" '
    d = f'> ./music_classification/models/{header_name}'
    cmd1 = a + b + c + d
    os.system(cmd1)
    os.system(f'cat ./music_classification/models/{model_name}.tflite | xxd -i      >> ./music_classification/models/{header_name}')
    a = 'echo "};"                              >> '
    b = f'./music_classification/models/{header_name}'
    cmd2 = a + b
    os.system(cmd2)
    model_h_size = os.path.getsize(f"./music_classification/models/{header_name}")
    print(f"Header file, model.h, is {model_h_size:,} bytes.")
    if _is_test_model:
        test(X_test, y_test, model, LE)
    # num_pass = 0
    # num_fail = 0
    # min_length = sample_rate
    # for i in range(80):
    #     # random_test = test_df.iloc[np.random.randint(0, len(test_df))]
    #     random_test = test_df.iloc[i]
    #     filename = "/Users/amadeus/Downloads/archive/Test_submission/Test_submission/"+random_test[0]
    #     print("filename=" + random_test[0])
    #     test_audio_raw, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    #     if _is_split == 1:
    #         test_intervals = librosa.effects.split(test_audio_raw, top_db=20)
    #         test_audio = librosa.effects.remix(test_audio_raw, intervals)
    #     elif _is_trim == 1:
    #         test_audio, label = librosa.effects.trim(test_audio_raw, top_db=30)
    #     else:
    #         test_audio = test_audio_raw
    #     chunked_test_data = []
    #     if len(test_audio) >= sample_rate:
    #         this_chunked_data = (chunk_data(test_audio, segment_length=min_length, overlap=0))
    #         for this_data in this_chunked_data:
    #             test_data = MFCC_Extractor(this_data, sample_rate)
    #             chunked_test_data.append(test_data)
    #     print('class of the random test: ', random_test[1])
    #     if len(chunked_test_data) == 0:
    #         continue
    #     chunked_test_data = np.array(chunked_test_data)
    #
    #     predicted_labels = np.argmax(model.predict(chunked_test_data), axis=-1)
    #     # print(predicted_labels)
    #
    #     likely_label = np.array(most_frequent(predicted_labels))
    #     likely_label = np.expand_dims(likely_label, axis=-1)
    #
    #     prediction_class = np.squeeze(LE.inverse_transform(likely_label), axis=-1)
    #     print('predicted class: ', prediction_class)
    #     if str(random_test[1])[6] == str(prediction_class)[6]:
    #         print('PASS')
    #         num_pass += 1
    #     else:
    #         print('FAIL')
    #         num_fail += 1
    # print(str(num_pass) + ',' + str(num_fail))


