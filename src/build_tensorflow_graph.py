
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

def new_convLayer(inputLayer, cnnArchitecture):
    paddingAlgorithm = 'SAME' if cnnArchitecture.toPadding else 'VALID'
    filterSpec = list(cnnArchitecture.filterSize) + [cnnArchitecture.numInputChannels, 
                                                     cnnArchitecture.numFilters]
    weights = new_weights(filterSpec)
    biases = new_biases(cnnArchitecture.numFilters)
    
    convLayer = tf.nn.conv2d(input = inputLayer, 
                             filter = weights, 
                             strides = [1, cnnArchitecture.strides, 
                                        cnnArchitecture.strides], 
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


# In[4]:


if __name__ == '__main__':
    cnnArchitecture = CNN_Architecture(numFilters = 20, 
                                       filterSize = (3, 3), 
                                       strides = 2, 
                                       toPadding = False, 
                                       useReLU = True,
                                       numInputChannels = 200,
                                       maxPoolingSize=(2, 2))

