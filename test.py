#!/usr/bin/env python

import tensorflow as tf
import numpy as np

d = np.zeros((1,2,3,4,2,7))

d[0][0][0][0][0] = np.array([1,2,3,4,5,6,7])
d[0][0][0][0][1] = np.array([1,2,3,0,5,6,7])

d = tf.convert_to_tensor(d, np.float32)

with tf.Session() as s:
    x = tf.reduce_max(d, axis=-2, keep_dims=True)
    print(x.eval())
