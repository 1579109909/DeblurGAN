import tensorflow as tf
import numpy as np


def Conv(name, x, filter_size, in_filters, out_filters, strides, padding):
    with tf.variable_scope(name):
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable('bias', [out_filters], tf.float32, initializer=tf.zeros_initializer())

        return tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding=padding) + bias


def Conv_transpose(name, x, filter_size, in_filters, out_filters, fraction=2, padding="SAME"):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [filter_size, filter_size, out_filters, in_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)

        return x


def instance_norm(x, BN_epsilon=1e-3):
    mean, variance = tf.nn.moments(x, axes=[1, 2])
    x = (x - mean) / ((variance + BN_epsilon) ** 0.5)
    return x

