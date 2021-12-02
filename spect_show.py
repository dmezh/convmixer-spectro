# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


spectrogram = np.load("spectrogramsSound8K/audio/fold1/99180-9-0-48.npy")
print(spectrogram.shape)

fig, ax = plt.subplots()
img = librosa.display.specshow(spectrogram,
                               y_axis='log',
                               x_axis='time',
                               ax=ax)

ax.set_title('Amplitude spectrogram street music')
fig.colorbar(img, ax=ax, format="%+2.0f dB")

fig.savefig('test.png')

