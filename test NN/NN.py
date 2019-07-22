from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#image stuff
from imutils import paths
import cv2
import os
import random

data = []
labels = []

test_data = []
test_labels = []

imagePaths = sorted(list(paths.list_images("images")))
testImagePaths = sorted(list(paths.list_images("test_images")))

random.seed(420)
random.shuffle(imagePaths)
random.shuffle(testImagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = keras.preprocessing.image.img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "open_hand" else 0
    labels.append(label)

for testPath in testImagePaths:
    image = cv2.imread(testPath)
    image = cv2.resize(image, (28, 28))
    test_data.append(image)

    label = testPath.split(os.path.sep)[-2]
    label = 1 if label == "open_hand" else 0
    test_labels.append(label)

x_train = np.array(data)
y_train = np.array(labels)

x_test = np.array(test_data)
y_test = np.array(test_labels)

model = keras.Sequential([
    keras.layers.Input((28, 28, 3)),
    keras.layers.Dense(35, activation=tf.nn.relu),
    keras.layers.Conv2D(16, (5, 5), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D((2, 2), 2),
    keras.layers.Dense(70, activation=tf.nn.relu),
    keras.layers.Conv2D(32, (5, 5), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D((2, 2), 2),
    keras.layers.Conv2D(32, (5, 5), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D((2, 2), 2),
    keras.layers.Dropout(0.22),
    keras.layers.Flatten(),
    keras.layers.Dense(115, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#print(data)
#print(labels)
print(x_train.shape)
print(y_train.shape)

model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=15, validation_data=(x_test, y_test))