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

data = []
labels = []

test_data = []
test_labels = []

imagePaths = paths.list_images("images")
testImagePaths = paths.list_images("test_images")

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
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.relu,),
    keras.layers.Dense(10, activation=tf.nn.relu),
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

model.fit(x_train, y_train, batch_size=2, epochs=5, validation_data=(x_test, y_test))