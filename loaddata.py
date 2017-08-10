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

    data = np.zeros((len(data_all), 96*96))
    labels = np.zeros((len(data_all), 1))
    for i, pic in enumerate(data_all['Image']):
        image1 = pic.split(' ')
        image1 = [int(x) for x in image1]
        data[i] = np.array(image1)

        labels[i] = (np.mean(data[i]) > 128)

    # print(data)
    # print(y_labels)



    reset_tf()

    x = tf.placeholder(tf.float32, [None, 96*96], name="features")
    y_label = tf.placeholder(tf.float32, [None, 1], name="labels")

    W = tf.Variable(tf.zeros([96*96, 1]), name="weights")
    b = tf.Variable(tf.zeros([1]), name="biases")
    y = tf.matmul(x, W) + b



    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_label))
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    predicted = tf.cast(tf.nn.sigmoid(y) > 0.5, np.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_label), np.float32))

    reset_vars()

    print(sess.run(W))

    for i in range(300):
        sess.run(train, feed_dict={x: data, y_label: labels})
        if i % 30 == 0:
            print sess.run([loss, accuracy], feed_dict={x: data, y_label: labels})

    # print("W = ")
    # print(sess.run(W))
    # print("b = ")
    # print(sess.run(b))
    # print("min W = ")
    # print(sess.run(np.min(W[0])))
    # print(sess.run(np.min(np.array(W).flatten())))
    # print("max W = ")
    # print(sess.run(np.max(np.array(W).flatten())))

    W = sess.run(W)
    # print(W)
    print('min W = ')
    print(np.min(W.flatten()))
    print(np.argmin(W.flatten()))
    print('max W = ')
    print(np.max(W.flatten()))
    print(np.argmax(W.flatten()))








if __name__ == "__main__":
    main()