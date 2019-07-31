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

predict_data = []

imagePaths = sorted(list(paths.list_images("images")))
testImagePaths = sorted(list(paths.list_images("test_images")))
predictImagePaths = sorted(list(paths.list_images("predict_images")))

random.seed(420)
random.shuffle(imagePaths)
random.shuffle(testImagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "open_hand" else 2 if label == "thumbs_up" else 3 if label == "thumbs_down" else 0
    labels.append(label)

for testPath in testImagePaths:
    image = cv2.imread(testPath)
    image = cv2.resize(image, (28, 28))
    test_data.append(image)

    label = testPath.split(os.path.sep)[-2]
    label = 1 if label == "open_hand" else 2 if label == "thumbs_up" else 3 if label == "thumbs_down" else 0
    test_labels.append(label)

for predictPath in predictImagePaths:
    image = cv2.imread(predictPath)
    image = cv2.resize(image, (28, 28))
    predict_data.append(image)

x_train = np.array(data)
y_train = np.array(labels)

x_test = np.array(test_data)
y_test = np.array(test_labels)

x_predict = np.array(predict_data)

model = keras.Sequential([
    keras.layers.Input((28, 28, 3)),
    keras.layers.Dense(65, activation=tf.nn.relu),
    keras.layers.Conv2D(16, (5, 5), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D((2, 2), 2),
    keras.layers.Dense(90, activation=tf.nn.relu),
    keras.layers.Conv2D(32, (5, 5), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D((2, 2), 2),
    keras.layers.Conv2D(64, (5, 5), padding="same", activation=tf.nn.relu),
    keras.layers.MaxPool2D((2, 2), 2),
    keras.layers.Dropout(0.22),
    keras.layers.Flatten(),
    keras.layers.Dense(135, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.softmax)
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

checkpoint_path = "training_32/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(x_train, y_train, batch_size=8, epochs=15, validation_data=(x_test, y_test), callbacks = [cp_callback])

class_names = ["Closed", "Open"]

def plot_value_array(i, predictions_array, true_label):
  predictions_array = predictions_array[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

img = predict_data[0]

img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

# plot_value_array(0, predictions_single, 1)
# plt.xticks(range(10), class_names, rotation=45)
# plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)