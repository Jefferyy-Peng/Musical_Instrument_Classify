import os
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.io import wavfile
import librosa
import librosa.display
import pickle
import fileinput
# # import IPython.display as ipd
#
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Dropout,Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ModelCheckpoint
# from datetime import datetime

def chunk_data(data, segment_length=50, overlap=0.5):
    # Calculate the number of samples per segment and the number of overlapping samples
    overlap_samples = int(overlap * segment_length)

    # Calculate the total number of segments
    total_segments = int(np.ceil(data.shape[0]/(segment_length - overlap_samples)))
    # Initialize an empty list to store the segments
    chunked_data = []
    for i in range(total_segments):
        # Calculate the start and end samples of the current segment
        start = int(i * (segment_length - overlap_samples))
        end = int(start + segment_length)

        # Extract the current segment from the audio data
        segment = data[start:end]

        # Append the segment to the list of segments
        if segment.shape[0] == segment_length:
            chunked_data.append(segment)
    return chunked_data


def MFCC_Extractor(audio, sample_rate):
    # for CNN
    # print(label)
    # fig,axs = plt.subplots(nrows=2, ncols=1)
    # librosa.display.waveshow(audio_raw, sr=sample_rate, ax=axs[0])
    # librosa.display.waveshow(audio, sr=sample_rate, ax=axs[1])
    # plt.show()
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=96)
    # mfcc_dif1 = librosa.feature.delta(mfcc)
    # mfcc_dif2 = librosa.feature.delta(mfcc_dif1)
    # mfcc_d1_d2 = np.concatenate([mfcc,mfcc_dif1,mfcc_dif2],axis=0)
    return mfcc


def preprocess_train_data(df, _is_trim=False, _is_split=False, verbose=False):
    chunked_data = []
    discarded_data = []
    num_drum = 0
    num_guitar = 0
    num_piano = 0
    num_violin = 0
    for index_num,row in tqdm(df.iterrows()):
        file_name = os.path.join(os.path.abspath('/Users/amadeus/Downloads/archive/Train_submission/Train_submission/'), str(row["FileName"]))
        final_class_label=row["Class"]
        audio_raw, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        if final_class_label == 'background_noise':
            audio = audio_raw
        else:
            min_length = sample_rate
            if _is_trim == True:
                audio, label = librosa.effects.trim(audio_raw, top_db=30)
            elif _is_split == True:
                intervals = librosa.effects.split(audio_raw, top_db=20)
                audio = librosa.effects.remix(audio_raw, intervals)
            else:
                audio = audio_raw
        # fig,axs = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True)
        # librosa.display.waveshow(audio_raw, sr=sample_rate, ax = axs[0])
        # librosa.display.waveshow(audio_remix, sr=sample_rate, ax = axs[1], offset=intervals[0][0]/sample_rate)
        # for interval in intervals:
        #     axs[0].vlines(interval[0]/sample_rate,-0.5,0.5,colors='r')
        #     axs[0].vlines(interval[1]/sample_rate,-0.5,0.5,colors='r')
        # plt.show()
        if len(audio) >= min_length:
            this_chunked_data = (chunk_data(audio, segment_length=min_length, overlap=0))
            num_chunks = len(this_chunked_data)
            if final_class_label == 'Sound_Drum':
                num_drum += num_chunks
            elif final_class_label == 'Sound_Guitar':
                num_guitar += num_chunks
            elif final_class_label == 'Sound_Piano':
                num_piano += num_chunks
            else:
                num_violin += num_chunks
            for this_data in this_chunked_data:
                data=MFCC_Extractor(this_data, sample_rate)
                chunked_data.append([str(row["FileName"]), data, final_class_label])
        else:
            print(index_num)
            discarded_data.append([str(row["FileName"]), audio, final_class_label])
    drum_concat = []
    guitar_concat = []
    piano_concat = []
    violin_concat = []
    bn_concat = []
    for con_data in discarded_data:
        if con_data[2] == 'Sound_Drum':
            drum_concat.append(con_data[1])
        elif con_data[2] == 'Sound_Guitar':
            guitar_concat.append(con_data[1])
        elif con_data[2] == 'Sound_Piano':
            piano_concat.append(con_data[1])
        elif con_data[2] == 'Sound_Violin':
            violin_concat.append(con_data[1])
        else:
            bn_concat.append(con_data[1])
    # process remaining guitar
    if len(guitar_concat) != 0:
        guitar_concat = np.concatenate(guitar_concat)
        guitar_chunked_data = chunk_data(guitar_concat, segment_length=min_length, overlap=0)
        for this_data in guitar_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar, ['concat', data, 'Sound_Guitar'])
            num_guitar += 1
    # process remaining drum
    if len(drum_concat) != 0:
        drum_concat = np.concatenate(drum_concat)
        drum_chunked_data = chunk_data(drum_concat, segment_length=min_length, overlap=0)
        for this_data in drum_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar + num_drum, ['concat', data, 'Sound_Drum'])
            num_drum += 1
    # process remaining piano
    if len(piano_concat) != 0:
        piano_concat = np.concatenate(piano_concat)
        piano_chunked_data = chunk_data(piano_concat, segment_length=min_length, overlap=0)
        for this_data in piano_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar + num_piano, ['concat', data, 'Sound_Piano'])
            num_piano += 1
    # process remaining violin
    if len(violin_concat) != 0:
        violin_concat = np.concatenate(violin_concat)
        violin_chunked_data = chunk_data(violin_concat, segment_length=min_length, overlap=0)
        violin_data = []
        for this_data in violin_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar + num_violin, ['concat', data, 'Sound_Violin'])
            num_violin += 1
    if verbose:
        print("num_drum = " + str(num_drum))
        print("num_guitar = " + str(num_guitar))
        print("num_piano = " + str(num_piano))
        print("num_violin = " + str(num_violin))
    extracted_features_df = pd.DataFrame(chunked_data,columns=['FileName','features','class'])
    return extracted_features_df

def preprocess_test_data(df, _is_trim=False, _is_split=False, verbose=False):
    chunked_data = []
    discarded_data = []
    num_drum = 0
    num_guitar = 0
    num_piano = 0
    num_violin = 0
    for index_num,row in tqdm(df.iterrows()):
        file_name = os.path.join(os.path.abspath('/Users/amadeus/Downloads/archive/Test_submission/Test_submission/'), str(row["FileName"]))
        final_class_label=row["Class"]
        audio_raw, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        if final_class_label == 'background_noise':
            audio = audio_raw
        else:
            min_length = sample_rate
            if _is_trim == True:
                audio, label = librosa.effects.trim(audio_raw, top_db=30)
            elif _is_split == True:
                intervals = librosa.effects.split(audio_raw, top_db=20)
                audio = librosa.effects.remix(audio_raw, intervals)
            else:
                audio = audio_raw
        # fig,axs = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True)
        # librosa.display.waveshow(audio_raw, sr=sample_rate, ax = axs[0])
        # librosa.display.waveshow(audio_remix, sr=sample_rate, ax = axs[1], offset=intervals[0][0]/sample_rate)
        # for interval in intervals:
        #     axs[0].vlines(interval[0]/sample_rate,-0.5,0.5,colors='r')
        #     axs[0].vlines(interval[1]/sample_rate,-0.5,0.5,colors='r')
        # plt.show()
        if len(audio) >= min_length:
            this_chunked_data = (chunk_data(audio, segment_length=min_length, overlap=0))
            num_chunks = len(this_chunked_data)
            if verbose:
                if final_class_label == 'Sound_Drum':
                    num_drum += num_chunks
                elif final_class_label == 'Sound_Guitar':
                    num_guitar += num_chunks
                elif final_class_label == 'Sound_Piano':
                    num_piano += num_chunks
                else:
                    num_violin += num_chunks
            for this_data in this_chunked_data:
                data=MFCC_Extractor(this_data, sample_rate)
                chunked_data.append([str(row["FileName"]), data, final_class_label])
        else:
            print(index_num)
            discarded_data.append([str(row["FileName"]), audio, final_class_label])
    drum_concat = []
    guitar_concat = []
    piano_concat = []
    violin_concat = []
    bn_concat = []
    for con_data in discarded_data:
        if con_data[2] == 'Sound_Drum':
            drum_concat.append(con_data[1])
        elif con_data[2] == 'Sound_Guitar':
            guitar_concat.append(con_data[1])
        elif con_data[2] == 'Sound_Piano':
            piano_concat.append(con_data[1])
        elif con_data[2] == 'Sound_Violin':
            violin_concat.append(con_data[1])
        else:
            bn_concat.append(con_data[1])
    # process remaining guitar
    if len(guitar_concat) != 0:
        guitar_concat = np.concatenate(guitar_concat)
        guitar_chunked_data = chunk_data(guitar_concat, segment_length=min_length, overlap=0)
        for this_data in guitar_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar, ['concat', data, 'Sound_Guitar'])
            num_guitar += 1
    # process remaining drum
    if len(drum_concat) != 0:
        drum_concat = np.concatenate(drum_concat)
        drum_chunked_data = chunk_data(drum_concat, segment_length=min_length, overlap=0)
        for this_data in drum_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar + num_drum, ['concat', data, 'Sound_Drum'])
            num_drum += 1
    # process remaining piano
    if len(piano_concat) != 0:
        piano_concat = np.concatenate(piano_concat)
        piano_chunked_data = chunk_data(piano_concat, segment_length=min_length, overlap=0)
        for this_data in piano_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar + num_piano, ['concat', data, 'Sound_Piano'])
            num_piano += 1
    # process remaining violin
    if len(violin_concat) != 0:
        violin_concat = np.concatenate(violin_concat)
        violin_chunked_data = chunk_data(violin_concat, segment_length=min_length, overlap=0)
        violin_data = []
        for this_data in violin_chunked_data:
            data = MFCC_Extractor(this_data, sample_rate)
            chunked_data.insert(num_guitar + num_violin, ['concat', data, 'Sound_Violin'])
            num_violin += 1
    if verbose:
        print("num_drum = " + str(num_drum))
        print("num_guitar = " + str(num_guitar))
        print("num_piano = " + str(num_piano))
        print("num_violin = " + str(num_violin))
    extracted_features_df = pd.DataFrame(chunked_data,columns=['FileName','features','class'])
    return extracted_features_df