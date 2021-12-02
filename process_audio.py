#!/usr/bin/env python

import glob
import librosa
import librosa.display
import numpy as np
import multiprocessing
import os
from tqdm import tqdm

PADDING_SIZE = 174

def save_spectrogram(audio_file, target_sr):
    audio_ar, audio_sr = librosa.load(audio_file, sr=target_sr)

    spectrogram = librosa.stft(audio_ar)
    mag_spectrogram = np.abs(spectrogram)

    padded_spectrogram = librosa.util.pad_center(mag_spectrogram, PADDING_SIZE, axis=1)

    fname = audio_file.split('/')[-1]
    label = int(fname.split('-')[1])

    return np.array([padded_spectrogram, label, fname], dtype=object)

def process_single_file(entry):
    curr_file = entry
    target_file = "spectrograms" + curr_file[5:-4] + ".npy"

    #tqdm.write(f'IN {curr_dir} : {curr_file} --> {target_file}')

    spect = save_spectrogram(curr_file, sr)
    np.save(target_file, spect)


sr = 22000

folds = ['fold1','fold2','fold3','fold4','fold5',
         'fold6','fold7','fold8','fold9','fold10']
dir = "UrbanSound8K/audio/"
save_dir = "spectro_padded/"

all_f = glob.glob(dir + '/*/*.wav')

for f in folds:
    os.makedirs(os.path.dirname(save_dir + f), exist_ok=True)

with multiprocessing.Pool() as pool:
    list(tqdm(pool.imap(process_single_file, all_f), total=len(all_f)))
