# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import rnn_cell
from tensorflow.python.framework import function

from tensor2tensor.layers.common_layers import layer_norm
from tensor2tensor.layers.common_attention import add_timing_signal_1d, multihead_self_attention_memory_efficient

initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN',
                                                                          uniform=False, dtype=tf.float32)


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.
    """

    def __init__(self, hidden_size, keep_prob, cell_type, name='RNNEncoder'):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.name = name
        if cell_type == 'rnn_lstm':
            cell = rnn_cell.LSTMCell
        elif cell_type == 'rnn_gru':
            cell = rnn_cell.GRUCell
        else:
            cell = rnn_cell.LSTMCell
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = cell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = cell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with tf.variable_scope(self.name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class QAEncoder(object):
    '''
    Encoder block in QANet from https://arxiv.org/abs/1804.09541
    '''
    def __init__(self, num_blocks, num_layers, num_heads, filters, kernel_size, \
                 keep_prob, input_mapping=False, name='QAEncoder'):
        self.name = name
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.filters = filters
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.input_mapping = input_mapping
        
    def build_graph(self, inputs, masks):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if self.input_mapping:
                inputs = tf.layers.conv1d(inputs, filters=self.filters, \
                                          kernel_size=1, padding='SAME', name='input_mapping')
            outputs = inputs
            for i in range(self.num_blocks):
                with tf.variable_scope('block{}'.format(i + 1)):
                    outputs = add_timing_signal_1d(outputs)
                    for j in range(self.num_layers):
                        with tf.variable_scope('conv{}'.format(j + 1)):
                            outputs = layer_dropout(
                                x=outputs, 
                                fn=lambda x:tf.nn.dropout(tf.layers.separable_conv1d(layer_norm(x, name='ln1'), filters=self.filters, kernel_size=self.kernel_size, padding='SAME', name='conv{}'.format(j + 1)), self.keep_prob), 
                                keep_prob=1 - (j + 1) / self.num_layers * (1 - self.keep_prob))
                    outputs = tf.nn.dropout(outputs + multihead_self_attention(layer_norm(outputs, name='ln2'), masks, self.num_heads), self.keep_prob)
                    res = outputs
                    outputs = layer_norm(outputs, name='ln3')
                    #outputs = tf.layers.conv1d(outputs, filters=self.filters, kernel_size=1, padding='SAME', kernel_initializer=initializer_relu(), name='ffn1')
                    outputs = tf.nn.relu(tf.layers.conv1d(outputs, filters=self.filters, kernel_size=1,  padding='SAME', kernel_initializer=initializer_relu(), name='ffn1'))
                    outputs = tf.layers.conv1d(outputs, filters=self.filters, kernel_size=1, padding='SAME', name='ffn2')
                    outputs = tf.nn.dropout(res + outputs, self.keep_prob)
            return outputs
        

class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self, name="SimpleSoftmaxLayer"):
        self.name = name

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with tf.variable_scope(self.name):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

       
        
class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size, name="BasicAttn"):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.name = name
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with tf.variable_scope(self.name):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

        
class BiDAFAttn(object):
    '''Module for BiDAF attention cell from https://arxiv.org/abs/1611.01603
    '''
    def __init__(self, keep_prob=1.0, name='BiDAFAttn'):
        self.name = name
        self.keep_prob = keep_prob
        
    def build_graph(self, c, c_mask, q, q_mask):
        with tf.variable_scope(self.name):
            S = trilinear_similarity(c, q)
            _, alpha = masked_softmax(S, tf.expand_dims(q_mask, 1), 2)
            a = tf.matmul(alpha, q)
            m = tf.reduce_max(S, axis=2)
            _, beta = masked_softmax(m, c_mask, 1)
            c_attn = tf.matmul(tf.expand_dims(beta, 1), c)
            output = tf.concat([c, a, tf.multiply(c, a), tf.multiply(c, c_attn)], -1)
            return output


def masked_softmax(logits, mask, axis=-1):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, axis)
    return masked_logits, prob_dist

def trilinear_similarity(c, q, name='trilinear_similarity'):
    '''
    Calculate trilinear similarity matrix from https://arxiv.org/abs/1611.01603
    '''
    c_shape, q_shape = c.get_shape().as_list(), q.get_shape().as_list()
    c_len, q_len, h = c_shape[1], q_shape[1], c_shape[2]
    with tf.variable_scope(name):
        w_c = tf.get_variable('w_c', [h, 1], dtype=tf.float32)
        w_q = tf.get_variable('w_q', [h, 1], dtype=tf.float32)
        w_cq = tf.get_variable('w_cq', [1, 1, h], dtype=tf.float32)
        bias = tf.get_variable('bias', [1, 1, 1], dtype=tf.float32, initializer=tf.zeros_initializer())
        S = tf.reshape(tf.matmul(tf.reshape(c, [-1, h]), w_c), [-1, c_len, 1]) \
        + tf.reshape(tf.matmul(tf.reshape(q, [-1, h]), w_q), [-1, 1, q_len]) \
        + tf.matmul(tf.multiply(c, w_cq), tf.transpose(q, perm=[0, 2, 1]))
        S = tf.add(S, bias)
        return S


def max_product_span(start_dist, end_dist):
    '''
    Find the answer span with maximum probability
    '''
    batch_size, input_len = start_dist.shape
    i = np.zeros(batch_size, dtype=np.int32)
    start_pos = np.zeros(batch_size, dtype=np.int32)
    end_pos = np.zeros(batch_size, dtype=np.int32)
    start_argmax = np.zeros(batch_size, dtype=np.int32)
    max_start_prob = np.zeros(batch_size, dtype=np.float32)
    max_product = np.zeros(batch_size, dtype=np.float32)
    
    while np.any(i < input_len):
        start_prob = start_dist[np.arange(batch_size), i]
        start_argmax = np.where(start_prob > max_start_prob, i, start_argmax)
        max_start_prob = np.where(start_prob > max_start_prob, start_prob, max_start_prob)
        
        end_prob = end_dist[np.arange(batch_size), i]
        new_product = max_start_prob * end_prob
        start_pos = np.where(new_product > max_product, start_argmax, start_pos)
        end_pos = np.where(new_product > max_product, i, end_pos)
        max_product = np.where(new_product > max_product, new_product, max_product)
        
        i += 1
        
    return start_pos, end_pos
              
                
def layer_dropout(x, fn, keep_prob):
    '''
    Layer dropout in https://arxiv.org/abs/1603.09382
    '''
    return tf.cond(tf.random_uniform([]) <= keep_prob,
                   lambda: x + fn(x) / keep_prob,
                   lambda: x)
    
def attention_bias_ignore_padding(mask):
    '''Create an bias tensor to be added to attention logits.
    
    Adapted from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    '''
    ret = (1 - tf.cast(mask, 'float')) * (-1e9)
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def multihead_self_attention(x, mask, num_heads, head_size=None, bias=True, epsilon=1e-6, forget=True,
                             name='multihead_self_attention'):
    '''Multihead scaled-dot-product self-attention.
    
    Includes layer norm.
    
    Returns multihead-self-attention(layer_norm(x))
    
    Adapted from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    '''
    if bias:
        bias = attention_bias_ignore_padding(mask)
    y = multihead_self_attention_memory_efficient(x=x, bias=bias, num_heads=num_heads, \
                                                  head_size=head_size, epsilon=epsilon, \
                                                  forget=forget, name=name)
    return y


def highway(x, num_outputs=None, num_layers=2, activation=tf.nn.relu, keep_prob=1.0, name='highway'):
    with tf.variable_scope(name):
        if num_outputs is None:
            num_outputs = x.shape.as_list()[-1]
        for i in range(num_layers):
            T = tf.layers.conv1d(x, num_outputs, kernel_size=1, padding='SAME', 
                                 activation=tf.sigmoid, name='gate{}'.format(i))
            H = tf.layers.conv1d(x, num_outputs, kernel_size=1, padding='SAME',
                                 activation=activation, name='output{}'.format(i))
            x = tf.nn.dropout(H * T + x * (1.0 - T), keep_prob)
    return x