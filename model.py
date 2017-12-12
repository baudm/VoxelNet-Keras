
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Layer, Input, Concatenate

import keras.backend as K

#m=v.mean(axis=-2, keepdims=True)
#v - m

class RepeatElements(Layer):

    def __init__(self, rep, axis=-1, **kwargs):
        super(RepeatElements, self).__init__(**kwargs)
        self.rep = rep
        self.axis = axis

    def call(self, inputs, **kwargs):
        return K.repeat_elements(inputs, self.rep, self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] *= self.rep
        return tuple(output_shape)


class ElementwiseMaxPool(Layer):

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=-2, keepdims=True)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-2] = 1
        return tuple(output_shape)


def vfe_block(x, cout, name=None):
    assert cout % 2 == 0
    if name is None:
        name = 'vfe'
    x = Dense(cout // 2, name=name + '_dense')(x)
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    max = ElementwiseMaxPool(name=name + '_maxpool')(x)
    max = RepeatElements(x.shape[-2].value, axis=-2, name=name + '_repeat')(max)
    x = Concatenate(name=name + '_concat')([max, x])
    return x


def make():
    x = Input((1, 2, 3, 4, 2, 7))
    y = vfe_block(x, 32, 'vfe_1')
    m = Model(x, y)
    return m
