
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np 


# In[2]:


class CNN_Architecture:
    def __init__(self, numFilters, filterSize, strides, toPadding, 
                useReLU, numInputChannels, maxPoolingSize=None):
        self.numFilters = numFilters
        self.filterSize = filterSize
        self.strides = strides
        self.maxPoolingSize = maxPoolingSize
        self.toPadding = toPadding
        self.useReLU = useReLU
        self.numInputChannels = numInputChannels


# In[3]:


def new_weights(shape):
    stdev = 0.1
    w = tf.Variable(tf.random_uniform(shape=shape, minval = -stdev, maxval = stdev))
    return w

def new_biases(length):
    stdev = 0.1
    b = tf.Variable(tf.random_uniform(shape=[length], minval = -stdev, maxval = stdev))
    return b

def new_convLayer(inputLayer, cnnArchitecture, name = "conv2d", stdev = 0.01):
    with tf.variable_scope(name):
        paddingAlgorithm = 'SAME' if cnnArchitecture.toPadding else 'VALID'
        filterSpec = list(cnnArchitecture.filterSize) + [cnnArchitecture.numInputChannels, 
                                                         cnnArchitecture.numFilters]
        weights = tf.get_variable('w', filterSpec, 
                          initializer = tf.truncated_normal_initializer(stddev=stdev))
        biases = tf.get_variable('bias', [cnnArchitecture.numFilters], 
                                  initializer = tf.constant_initializer(0.0))

        convLayer = tf.nn.conv2d(input = inputLayer, 
                                 filter = weights, 
                                 strides = [1, cnnArchitecture.strides, 
                                            cnnArchitecture.strides, 1], 
                                 padding = paddingAlgorithm)
        convLayer = convLayer + biases

        if cnnArchitecture.maxPoolingSize:
            steps = [1, cnnArchitecture.maxPoolingSize[0], 
                        cnnArchitecture.maxPoolingSize[1], 1]
            convLayer = tf.nn.max_pool(value=convLayer, 
                                       ksize=steps, 
                                       strides = steps, 
                                       padding = paddingAlgorithm)
        if cnnArchitecture.useReLU: convLayer = tf.nn.relu(convLayer)

        return convLayer, weights

def new_dconvLayer(inputLayer, cnnArchitecture, outputShape, 
                   name = "dconv2d", stdev = 0.01):
    with tf.variable_scope(name):
        paddingAlgorithm = 'SAME' if cnnArchitecture.toPadding else 'VALID'
        filterSpec = list(cnnArchitecture.filterSize) + [cnnArchitecture.numInputChannels, 
                                                         cnnArchitecture.numFilters]
        weights = tf.get_variable('w', filterSpec, 
                          initializer = tf.truncated_normal_initializer(stddev=stdev))
        biases = tf.get_variable('bias', [outputShape[-1]], 
                                  initializer = tf.constant_initializer(0.0))

        convLayer = tf.nn.conv2d_transpose(inputLayer, 
                                           output_shape=outputShape,
                                           filter = weights, 
                                           strides = [1, cnnArchitecture.strides, 
                                                        cnnArchitecture.strides, 1], 
                                           padding = paddingAlgorithm)
        convLayer = convLayer + biases

#         if cnnArchitecture.maxPoolingSize:
#             steps = [1, cnnArchitecture.maxPoolingSize[0], 
#                         cnnArchitecture.maxPoolingSize[1], 1]
#             convLayer = tf.nn.max_pool(value=convLayer, 
#                                        ksize=steps, 
#                                        strides = steps, 
#                                        padding = paddingAlgorithm)
        if cnnArchitecture.useReLU: convLayer = tf.nn.relu(convLayer)

        return convLayer

def flattenLayer(layer):
    """
    [width, height, numFilters]
    """
    shape = layer.get_shape()
    # shape = [#imgs, height, width, numFilters]
    numAttrs = shape[1:].num_elements()
    
    layer_flat = tf.reshape(layer, shape=[-1, numAttrs])
    return layer_flat, numAttrs

def new_fcLayer(inputLayer, inputChannels, outputChannels, useReLU=True, 
                name="fc", stdev=0.01):
    with tf.variable_scope(name):
        weights = tf.get_variable('w', shape=[inputChannels, outputChannels], 
                                initializer=tf.random_normal_initializer(stddev=stdev))
        biases = tf.get_variable('bias', shape=[outputChannels], 
                                 initializer=tf.constant_initializer(0.0))
        layer = tf.matmul(inputLayer, weights) + biases
        if useReLU:
            layer = tf.nn.relu(layer)
        return layer
    
def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


# In[4]:


if __name__ == '__main__':
    cnnArchitecture = CNN_Architecture(numFilters = 20, 
                                       filterSize = (3, 3), 
                                       strides = 2, 
                                       toPadding = False, 
                                       useReLU = True,
                                       numInputChannels = 200,
                                       maxPoolingSize=(2, 2))

