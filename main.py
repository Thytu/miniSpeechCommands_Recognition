import conv2d
import model_process
import dataset_loader

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import wavfile

NEED_TO_CREATE_DATASET = False
NEED_TO_CREATE_H5 = False

if NEED_TO_CREATE_DATASET:
    dataset_loader.create_dataset("WAV_mini_speech_commands/")


if NEED_TO_CREATE_H5:
    all_set = dataset_loader.load_dataset("IMG_mini_speech_commands/", 64, 64)

    dataset, labels = dataset_loader.join_sets(all_set)
    train_set, train_labels, test_set, test_labels = dataset_loader.split_dataset(dataset, labels, 0.8)

    dataset_loader.create_h5_dataset(train_set, train_labels, test_set, test_labels)

train_set, train_labels, test_set, test_labels = dataset_loader.load_h5_dataset()

BATCH_SIZE = 128
# ---------- MAX ----------
# 32: 57% train - 44% test
# 64: 78% train - 69% test
# 128: 88% train - 71% test

train_set, train_labels = dataset_loader.create_batch(train_set, train_labels, BATCH_SIZE)
test_set, test_labels = dataset_loader.create_batch(test_set, test_labels, BATCH_SIZE)

CNN = conv2d.CNN()

EPOCHS = 20
LEARNING_RATE = 0.001


optimizer = optim.Adam(CNN.parameters(), lr=LEARNING_RATE)

training_losses = []
testing_losses = []

training_accuracies = []
testing_accuracies = []

for epoch in range(EPOCHS):
    training_loss, training_accuracy = model_process.train(CNN, optimizer, train_set, train_labels, batch_size=BATCH_SIZE)
    testing_loss, testing_accuracy = model_process.test(CNN, test_set, test_labels, batch_size=BATCH_SIZE)

    training_losses.append(training_loss)
    training_accuracies.append(training_accuracy)
    testing_losses.append(testing_loss)
    testing_accuracies.append(testing_accuracy)

    print(f"Epoch: {epoch + 1} accuracy: {training_accuracy:.2f} loss: {training_loss:.3f}", end="\t")
    print(f'Validation: Accuracy: {testing_accuracy:.2f} loss: {testing_loss:.3f}')


plt.plot(list(range(1, len(training_losses)+1)), training_losses, color='blue')
plt.plot(list(range(1, len(testing_losses)+1)), testing_losses, color='red')

plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Loss')
plt.show()


plt.plot(list(range(1, len(training_accuracies)+1)), training_accuracies, color='blue')
plt.plot(list(range(1, len(testing_accuracies)+1)), testing_accuracies, color='red')

plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Accuracy')
plt.show()