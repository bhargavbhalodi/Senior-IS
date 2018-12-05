# Bhargav Bhalodi
# Tensorflow model for object detection

# Help taken from the tutorial
# information given at
# https://www.tensorflow.org/tutorials/keras/basic_classification

# Help and code taken from
# Rajalingappaa Shanmugamani. Deep Learning for Computer Vision: Expert
# techniques to train advanced neural networks using TensorFlow and Keras.
# Packt Publishing - ebooks Account, Birmingham Mumbai, January 2018.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt



cifar100_dataset = keras.datasets.cifar10

(img_train, lab_train), (img_test, lab_test) = cifar100_dataset.load_data()

class_names = lab_train

img_train = img_train/255.0
img_test = img_test/255.0

plt.figure(figsize=(10,10))

for i in range(25):
    # print("attempting to display ... ", i)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_train[i], cmap=plt.cm.binary)
    # plt.xlabel(class_names[lab_train[i]])
    plt.xlabel(lab_train[i])
plt.show()

print("\n\n", img_train.shape)
# input_shape = img_train.shape
print("\n", lab_train.shape, "\n")
num_outputs = 10

# defining the YOLO model
#model = keras.Sequential([
#    keras.layers.Conv2D(input_shape=(28, 28, 3), filters=64,
#                        kernel_size=7,
#                        strides=2,
#                        padding="same"),
#    keras.layers.LeakyReLU(),
#    keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")
#])

#model.add(keras.layers.Conv2D(filters=192, kernel_size=3))
#model.add(keras.layers.LeakyReLU())
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=2,
# padding="same"))

# model.add(keras.layers.Conv2D(filters=128, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=256, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=256, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=3))
# model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same"))
#
# model.add(keras.layers.Conv2D(filters=256, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=256, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=256, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=256, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))
#
# model.add(keras.layers.Conv2D(filters=512, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=512, kernel_size=1))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3, strides=2, padding="same"))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3))
# model.add(keras.layers.LeakyReLU())
# model.add(keras.layers.Conv2D(filters=1024, kernel_size=3))
# model.add(keras.layers.LeakyReLU())

input_shape = img_train.shape[1:]
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape=input_shape,
                              filters=64,
                              kernel_size=3, strides=2, padding="same"))
model.add(keras.layers.LeakyReLU())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=512))
model.add(keras.layers.LeakyReLU())
model.add(keras.layers.Dense(units=4096))
model.add(keras.layers.LeakyReLU())
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=num_outputs, activation=tf.nn.softmax))

# compiles the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fits the model to training data
print(img_train)
model.fit(img_train, lab_train,
          epochs=1)

# evaluates the model against test data
test_loss, test_acc = model.evaluate(img_test, lab_test)
print('Test accuracy:', test_acc)
predictions = model.predict(img_test)


# plots examples from test data and their respective predictions
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], \
                                         img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  #print("trying to plot")
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, lab_test, img_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, lab_test)


plt.show()