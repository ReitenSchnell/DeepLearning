from skimage.data import astronaut
from scipy.misc import imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import data_sources.image_utils


def linear(m_input, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or 'linear'):
        W = tf.get_variable('W', shape=[n_input, n_output], initializer=tf.random_normal_initializer(0.0, 0.1))
        b = tf.get_variable('b', shape=[n_output], initializer=tf.constant_initializer())
        h = tf.matmul(m_input, W) + b
        if activation is not None:
            h = activation(h)
        return h


def distance(p1, p2):
    return tf.abs(p1 - p2)


def read_image():
    arr = astronaut()
    img = imresize(arr, (64, 64))
    xs = []
    ys = []
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])

    xs = np.array(xs)
    ys = np.array(ys)
    xs = (xs - np.mean(xs)) / np.std(xs)
    print(xs.shape, ys.shape)
    return xs, ys, img


def train_network(img, xs, ys):
    x = tf.placeholder(tf.float32, [None, 2])
    y = tf.placeholder(tf.float32, [None, 3])
    n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]
    current_input = x
    for layer in range(1, len(n_neurons)):
        current_input = linear(current_input, n_neurons[layer - 1], n_neurons[layer],
                               activation=tf.nn.relu if layer + 1 < len(n_neurons) else None,
                               scope='layer_' + str(layer))
    y_pred = current_input

    cost = tf.reduce_mean(tf.reduce_sum(distance(y_pred, y), 1))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    n_iterations = 500
    batch_size = 50
    images = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            indexes = np.random.permutation(range(len(xs)))
            n_batches = len(indexes) // batch_size
            for batch_i in range(n_batches):
                index_i = indexes[batch_i * batch_size: (batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={x: xs[index_i], y: ys[index_i]})
            training_cost = sess.run(cost, feed_dict={x: xs, y: ys})
            print(it_i, training_cost)
            if (it_i + 1) % 20 == 0:
                ys_pred = y_pred.eval(feed_dict={x: xs}, session=sess)
                seen_img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
                images.append(seen_img)
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                break
            prev_training_cost = training_cost
        data_sources.image_utils.build_gif(images)


xs, ys, img = read_image()
train_network(img, xs, ys)
