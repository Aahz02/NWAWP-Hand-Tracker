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

predict_data = []

predictImagePaths = sorted(list(paths.list_images("predict_images")))

for predictPath in predictImagePaths:
    image = cv2.imread(predictPath)
    image = cv2.resize(image, (28, 28))
    predict_data.append(image)

x_predict = np.array(predict_data)

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

model.summary()

checkpoint_path = "training_3/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)

predictions = model.predict(x_predict)

readable_predicts = []

for prediction in predictions:
    readable_predicts.append(np.argmax(prediction))

print(predictImagePaths)
print(readable_predicts)