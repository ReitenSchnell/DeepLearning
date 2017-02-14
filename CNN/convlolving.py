import os
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.data


def gabor_kernel(mean, sigma, k_size):
    x = tf.linspace(-3.0, 3.0, k_size)
    z = tf.exp(tf.neg(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415)))
    z_2d = tf.matmul(tf.reshape(z, tf.pack([k_size, 1])), tf.reshape(z, tf.pack([1, k_size])))
    ys = tf.sin(x)
    ys = tf.reshape(ys, tf.pack([k_size, 1]))
    ones = tf.ones(tf.pack([1, k_size]))
    wave = tf.matmul(ys, ones)
    kernel = tf.mul(wave, z_2d)
    kernel_4d = tf.reshape(kernel, tf.pack([k_size, k_size, 1, 1]))
    return kernel_4d


def gaussian_kernel(mean, sigma, k_size):
    x = tf.linspace(-3.0, 3.0, k_size)
    z = tf.exp(tf.neg(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415)))
    z_2d = tf.matmul(tf.reshape(z, tf.pack([k_size, 1])), tf.reshape(z, tf.pack([1, k_size])))
    kernel_4d = tf.reshape(z_2d, tf.pack([k_size, k_size, 1, 1]))
    return kernel_4d


def convolve_image(image):
    img = tf.placeholder(tf.float32, shape=[None, None])
    img_3d = tf.expand_dims(img, 2)
    img_4d = tf.expand_dims(img_3d, 0)
    mean = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    k_size = tf.placeholder(tf.int32)
    kernel = gabor_kernel(mean, sigma, k_size)
    convolved = tf.nn.conv2d(img_4d, kernel, strides=[1, 1, 1, 1], padding='SAME')
    convolved_img = convolved[0, :, :, 0]
    with tf.Session() as sess:
        res = sess.run(convolved_img, feed_dict={img: image, sigma: 1.0, mean: 0.0, k_size: 100})
        plt.imshow(res, cmap='gray')


convolve_image(skimage.data.camera())
plt.show()
