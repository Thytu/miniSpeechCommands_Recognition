import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from os import listdir
from scipy.io import wavfile
from sklearn.utils import shuffle

def setUpPlot_noBlankSpace():
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

def createSpecFromWav(inPath, outPath):
    samplingFrequency, signalData = wavfile.read(inPath)

    plt.subplot(211)
    plt.ylim(-20_000, 20_000)
    plt.xlim(0, 16_000)
    plt.axis('off')
    plt.plot(signalData)

    plt.subplot(212)
    plt.ylim(0, 8_000)
    plt.xlim(0, 1)
    plt.axis('off')
    plt.specgram(signalData,Fs=samplingFrequency)

    plt.savefig(outPath)
    plt.clf()

def create_dataset(dataset_path):
    setUpPlot_noBlankSpace()
    for folder in listdir(dataset_path):
        print(folder)
        for f in listdir(dataset_path + folder):
            createSpecFromWav(dataset_path + folder + "/" + f, dataset_path.replace("WAV", "IMG") + folder + "/" + f.replace("wav", "png"))


def load_dataset(dataset_path, width, height):
    all_set = {}

    for folder in listdir(dataset_path):
        folder_set = []
        print(folder)
        for f in listdir(dataset_path + folder):
            folder_set.append(np.asarray(Image.open(dataset_path + folder + "/" + f).resize((width, height)), dtype=np.int).reshape(-1, width, height))
        all_set[folder] = folder_set
    return all_set

def create_batch(dataset, labels, batch_size):
    dataset_result = []
    labels_result = []
    data_batch = []
    label_batch = []

    for index in range(len(dataset)):
        if len(data_batch) == batch_size:
            dataset_result.append(data_batch)
            labels_result.append(label_batch)
            data_batch = []
            label_batch = []
        else:
            data_batch.append(dataset[index])
            label_batch.append(labels[index])

    return torch.Tensor(dataset_result), torch.tensor(labels_result, dtype=torch.int64)

def join_sets(all_set):
    left_set = all_set["left"]
    left_labels = [0] * len(left_set)

    right_set = all_set["right"]
    right_labels = [1] * len(right_set)

    up_set = all_set["up"]
    up_labels = [2] * len(up_set)

    down_set = all_set["down"]
    down_labels = [3] * len(down_set)

    yes_set = all_set["yes"]
    yes_labels = [4] * len(yes_set)

    no_set = all_set["no"]
    no_labels = [5] * len(no_set)

    go_set = all_set["go"]
    go_labels = [6] * len(go_set)

    stop_set = all_set["stop"]
    stop_labels = [7] * len(stop_set)

    dataset = left_set + right_set + up_set + down_set + yes_set + no_set + go_set + stop_set
    labels = left_labels + right_labels + up_labels + down_labels + yes_labels + no_labels + go_labels + stop_labels

    return shuffle(np.array(dataset, dtype=np.int32), np.array(labels, dtype=np.int32))

def split_dataset(dataset, labels, train_ratio):
    return dataset[:int(len(dataset) * train_ratio)], labels[:int(len(labels) * train_ratio)], dataset[int(len(dataset) * train_ratio):], labels[int(len(labels) * train_ratio):]

def create_h5_dataset(train_set, train_labels, test_set, test_labels):
    file_set = h5py.File('dataset.hdf5', 'w')

    file_set.create_dataset("train_set", data=train_set)
    file_set.create_dataset("train_labels", data=train_labels)
    file_set.create_dataset("test_set", data=test_set)
    file_set.create_dataset("test_labels", data=test_labels)

def load_h5_dataset():
    dataset = h5py.File('dataset.hdf5', 'r')

    train_set = np.array(dataset["train_set"], dtype=np.int)
    train_labels = np.array(dataset["train_labels"], dtype=np.int)
    test_set = np.array(dataset["test_set"], dtype=np.int)
    test_labels = np.array(dataset["test_labels"], dtype=np.int)

    return train_set, train_labels, test_set, test_labels