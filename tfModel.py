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


cifar100_dataset = keras.datasets.fashion_mnist

(img_train, lab_train), (img_test, lab_test) = cifar100_dataset.load_data()

# class_names = lab_train

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


# defines a pooling layer
def pooling_layer(input_layer, pool_size=[2, 2], strides=2, padding='valid'):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides,
        padding=padding
    )
    # add_variable_summary(layer, 'pooling')
    return layer

# defines a convolution layer
def convolution_layer(input_layer, filters, kernel_size=[3, 3], padding='valid',
                      activation=tf.nn.leaky_relu):
    layer = tf.layers.conv2d(
        inputs=input_layer,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        # weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        # weights_regularizer=tf.contrib.layers.l2_regularizer(0.0005)
    )
    # add_variable_summary(layer, 'convolution')
    return layer

# defines a dense layer
def dense_layer(input_layer, units, activation=tf.nn.leaky_relu):
    layer = tf.layers.dense(
        inputs=input_layer,
        units=units,
        activation=activation,
        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0005)
    )
    # add_variable_summary(layer, 'dense')
    return layer


# defines the YOLO model
#yolo = tf.pad(img_train, np.array([[0, 0], [3, 3], [3,3], [0, 0]]),
#              name='pad_1')
yolo = convolution_layer(img_train, 64, 7, 2)
yolo = pooling_layer(yolo, [2, 2], 2, 'same')
yolo = convolution_layer(yolo, 192, 3)
yolo = pooling_layer(yolo, 2, 'same')
yolo = convolution_layer(yolo, 128, 1)
yolo = convolution_layer(yolo, 256, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = pooling_layer(yolo, 2, 'same')
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 256, 1)
yolo = convolution_layer(yolo, 512, 3)
yolo = convolution_layer(yolo, 512, 1)
yolo = convolution_layer(yolo, 1024, 3)
yolo = pooling_layer(yolo, 2)
yolo = convolution_layer(yolo, 512, 1)
yolo = convolution_layer(yolo, 1024, 3)
yolo = convolution_layer(yolo, 512, 1)
yolo = convolution_layer(yolo, 1024, 3)
yolo = convolution_layer(yolo, 1024, 3)
yolo = tf.pad(yolo, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
yolo = convolution_layer(yolo, 1024, 3, 2)
yolo = convolution_layer(yolo, 1024, 3)
yolo = convolution_layer(yolo, 1024, 3)
yolo = tf.transpose(yolo, [0, 3, 1, 2])
yolo = tf.layers.flatten(yolo)
yolo = dense_layer(yolo, 512)
yolo = dense_layer(yolo, 4096)

dropout_bool = tf.placeholder(tf.bool)
yolo = tf.layers.dropout(
        inputs=yolo,
        rate=0.4,
        training=dropout_bool
    )
yolo = dense_layer(yolo, 10, None)


yolo.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

yolo.fit(img_train, lab_train, epochs=5)

test_loss, test_acc = yolo.evaluate(img_test, lab_test)

print('Test accuracy:', test_acc)

predictions = yolo.predict(img_test)


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