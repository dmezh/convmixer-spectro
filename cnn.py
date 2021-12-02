import glob
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

CLASS_NAME_LUT = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music',
}

def read_npy_file(item):
    data, label, fname = np.load(item.numpy(), allow_pickle=True)
#    print(data.shape)
    return data, label, fname

def main():
    fold1 = glob.glob("spectrogramsSound8K/audio/fold1/*.npy")
#    print(fold1)

    dataset = tf.data.Dataset.from_tensor_slices(fold1)

    # remove later!
    dataset = dataset.shuffle(10000)

    dataset = dataset.map(
            lambda item: tf.py_function(read_npy_file, [item], [tf.float32, tf.int8, tf.string]))

    image, label, fname = next(dataset.as_numpy_iterator())
#    print(image.shape)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(image,
                                   y_axis='log',
                                   x_axis='time',
                                   ax=ax)

    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f'{fname}\nClass: {CLASS_NAME_LUT[label]}')
    fig.savefig('test.png')

    #for e in dataset.as_numpy_iterator():
    #`    pass

if __name__ == "__main__":
    main()
