"""
This script performs logistic regression on a set of images.
We define the labels as whether the first pixel in the image is above
of below 128. This results in a pretty good predictor.
When the labels are the mean of the pixels, the prediction is not very
good.
Version 1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf


global sess


def plot_faces(data):
    for i in range(10):
        image1 = data['Image'][i]
        image1 = image1.split(' ')
        image1 = [int(x) for x in image1]
        image1 = np.array(image1)
        plt.imshow(image1.reshape((96, 96)), cmap=plt.cm.gray)
        plt.savefig('images/image'+ str(i) +'.jpg')

def reset_vars():
    sess.run(tf.global_variables_initializer())

def reset_tf():
    global sess
    if sess:
        sess.close()
    tf.reset_default_graph()
    sess = tf.Session()

def main():
    global sess
    sess = None

    data_all = pd.read_csv('training_sample.csv')
    # plot_faces(data)
    # print(data_all.head())
    output_size = 2
    #train_n = len(data_all)
    train_n = 500


    data = np.zeros((train_n, 96*96))
    labels = np.zeros((train_n, output_size))
    # print len(labels)
    for i, pic in enumerate(data_all['Image'][0:train_n]):
        image1 = pic.split(' ')
        image1 = [int(x) for x in image1]
        data[i] = np.array(image1)
        # print("data size = " + data.shape)

    #        labels[i] = np.float32((np.mean(data[i]) > 128))
        labels[i][0] = data_all.ix[i]['left_eye_center_x']
        labels[i][1] = data_all.ix[i]['left_eye_center_y']
        # labels[i] = np.float32(data[i][0] > 128)
        # print(data[i][0])
        # labels[i] = data_all['nose_tip_x'][i]
    data /= 255.  # Normalize the data between [0,1]

    # print(data)
    # print(labels)

    index_train = range(int(2*train_n/3))
    index_test = range(int(2*train_n/3), train_n)
    # index_train = range(len(data_all)-1)
    # index_test = [len(data_all)-1]

    reset_tf()

    hidden_size = 500


    x = tf.placeholder(tf.float32, [None, 96*96], name="features")
    y_label = tf.placeholder(tf.float32, [None, output_size], name="labels")

    W1 = tf.Variable(tf.random_normal([96*96, hidden_size], seed=42), name="weight1")
    b1 = tf.Variable(tf.zeros([hidden_size]), name="bias1")

    hidden = tf.nn.relu(tf.matmul(x, W1) + b1, name="hidden")
    # hidden = tf.matmul(x, W1) + b1

    W2 = tf.Variable(tf.random_normal([hidden_size, output_size], seed=24), name="weight2")
    b2 = tf.Variable(tf.zeros([1]), name="bias2")

    y = tf.matmul(hidden, W2) + b2

    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_label))
    loss = tf.reduce_mean(tf.square(y - y_label))
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # predicted = tf.cast(tf.nn.sigmoid(y) > 0.5, np.float32)
    accuracy = tf.reduce_mean(tf.square(y - y_label))

    reset_vars()

    print "loss, accuracy during training"
    for i in range(100):
        sess.run(train, feed_dict={x: data[index_train], y_label: labels[index_train]})
        if i % 30 == 0:
            print sess.run([loss, accuracy], feed_dict={x: data[index_train], y_label: labels[index_train]})
            # print ('y, y_label')
            # print sess.run([y, y_label])
        elif i == 1:
            print "Starting performance:" , sess.run([loss, accuracy], feed_dict={x: data[index_train], y_label: labels[index_train]})

    print "loss, accuracy on test data"
    print sess.run([loss, accuracy], feed_dict={x: data[index_test], y_label: labels[index_test]})
    # print sess.run(predicted)
    # print index_test
    # print labels[index_test]
    # print len(labels)
    print  "W2 = "
    print sess.run(W2)
    print "b = "
    print(sess.run(b2))
    # print("min W = ")
    # print(sess.run(np.min(W[0])))
    # print(sess.run(np.min(np.array(W).flatten())))
    # print("max W = ")
    # print(sess.run(np.max(np.array(W).flatten())))

#    W = sess.run(W)
    # print(W)
#    print('min W = ')
#    print(np.min(W.flatten()))
#    print(np.argmin(W.flatten()))
#    print('max W = ')
#    print(np.max(W.flatten()))
#    print(np.argmax(W.flatten()))

if __name__ == "__main__":
    main()