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
from matplotlib import pyplot

global sess


def plot_faces(data):
    for i in range(10):
        image1 = data['Image'][i]
        image1 = image1.split(' ')
        image1 = [int(x) for x in image1]
        image1 = np.array(image1)
        plt.imshow(image1.reshape((96, 96)), cmap=plt.cm.gray)
        plt.savefig('images/image'+ str(i) +'.jpg')

def plot_sample(x, y, axis):
    """
    Plots a single sample image with keypoints on top.

    Parameters
    ----------
    x     :
            Image data.
    y     :
            Keypoints to plot.
    axis  :
            Plot over which to draw the sample.
    """
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

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
    num_keypoints = 10
    #train_n = len(data_all)
    train_n = 200

    # Some columns have NA values.
    #data_all = data_all.dropna()

    # Normalize the non-picture parts (labels) to [-1,1]
    data_all[data_all.columns[:-1]] = (data_all[data_all.columns[:-1]] - 48) / 48
    data = np.zeros((train_n, 96*96))
    labels = np.zeros((train_n, num_keypoints))
    # print len(labels)
    for i, pic in enumerate(data_all['Image'][0:train_n]):
        image1 = pic.split(' ')
        image1 = [int(x) for x in image1]
        data[i] = np.array(image1)
        # print("data size = " + data.shape)

        labels[i][0] = data_all.ix[i]['left_eye_center_x']
        labels[i][1] = data_all.ix[i]['left_eye_center_y']
        labels[i][2] = data_all.ix[i]['right_eye_center_x']
        labels[i][3] = data_all.ix[i]['right_eye_center_y']
        labels[i][4] = data_all.ix[i]['nose_tip_y']
        labels[i][5] = data_all.ix[i]['nose_tip_x']
        labels[i][6] = data_all.ix[i]['mouth_right_corner_x']
        labels[i][7] = data_all.ix[i]['mouth_right_corner_y']
        labels[i][8] = data_all.ix[i]['mouth_left_corner_x']
        labels[i][9] = data_all.ix[i]['mouth_left_corner_y']


        # labels[i] = np.float32(data[i][0] > 128)
        # print(data[i][0])
        # labels[i] = data_all['nose_tip_x'][i]
    data /= 255.  # Normalize the data between [0,1]

    index_train = range(int(2*train_n/3))
    index_test = range(int(2*train_n/3), train_n)


    reset_tf()

    hidden_size = 300

    x = tf.placeholder(tf.float32, [None, 96*96], name="features")
    y_label = tf.placeholder(tf.float32, [None, num_keypoints], name="labels")

    W1 = tf.get_variable('weight1',shape=[96*96, hidden_size],
                    initializer=tf.contrib.layers.xavier_initializer(seed=42))
    b1 = tf.get_variable("bias1", shape=[hidden_size], initializer=tf.constant_initializer(0.0))

    #hidden = tf.nn.relu(tf.matmul(x, W1) + b1 , name="hidden")
    hidden = tf.nn.sigmoid(tf.matmul(x, W1) + b1 , name="hidden")

    # hidden = tf.matmul(x, W1) + b1

    W2 = tf.get_variable("weight2", shape=[hidden_size, num_keypoints],
                         initializer=tf.contrib.layers.xavier_initializer(seed=42))
    b2 = tf.get_variable("bias2", shape=[num_keypoints], initializer=tf.constant_initializer(0.0))

    y = tf.matmul(hidden, W2) + b2

    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_label))
    loss = tf.reduce_mean(tf.square(y - y_label))

    #train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    # predicted = tf.cast(tf.nn.sigmoid(y) > 0.5, np.float32)
    # Optimizer.
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=0.01,
        momentum=0.9,
        use_nesterov=True
    ).minimize(loss)


    reset_vars()

    print "loss, accuracy during training"
    for i in range(500):
        sess.run([optimizer], feed_dict={x: data[index_train], y_label: labels[index_train]})
        if i % 10 == 0:
            #print sess.run([hidden[0]], feed_dict={x: data[index_test][0:2]})
            print sess.run([loss], feed_dict={x: data[index_train], y_label: labels[index_train]})
            # print ('y, y_label')
            # print sess.run([y, y_label])
        if i == 1:
            print "Starting performance:" , sess.run([loss], feed_dict={x: data[index_train], y_label: labels[index_train]})

    print sess.run([y], feed_dict={x: data[index_test][0:2]})

    pred = []
    [p_batch] = sess.run([y], feed_dict={ x: data[0:100]})

    pred.extend(p_batch)
    fig = pyplot.figure(figsize=(2, 2))

    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
        plot_sample(data[i], pred[i], ax)
    pyplot.show()

if __name__ == "__main__":
    main()