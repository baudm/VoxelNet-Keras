
from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Layer, Input, Concatenate,\
    Conv3D, ZeroPadding3D, Permute, Reshape, Conv2D, ZeroPadding2D, Conv2DTranspose

import keras.backend as K

#m=v.mean(axis=-2, keepdims=True)
#v - m


T_AXIS = -2


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

    def get_config(self):
        config = {'rep': self.rep, 'axis': self.axis}
        base_config = super(RepeatElements, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ElementwiseMaxPool(Layer):

    def __init__(self, keepdims=False, **kwargs):
        super(ElementwiseMaxPool, self).__init__(**kwargs)
        self.keepdims = keepdims

    def call(self, inputs, **kwargs):
        return K.max(inputs, axis=T_AXIS, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:T_AXIS]
        if self.keepdims:
            output_shape += (1,)
        output_shape += (input_shape[T_AXIS + 1],)
        return output_shape

    def get_config(self):
        config = {'keepdims': self.keepdims}
        base_config = super(ElementwiseMaxPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def fcn_block(x, units, name='fcn'):
    # Hack for Tensorflow because it can't handle bias terms which have more than 5D
    shape = x._keras_shape
    d = shape[-5]
    h = shape[-4]
    s = shape[-3:]
    shape = (d*h,) + s
    x = Reshape(shape)(x)
    x = Dense(units, name=name + '_dense')(x)
    # Revert back to actual shape
    x = Reshape((d, h, s[0], s[1], units))(x)
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def vfe_block(x, cout, name='vfe'):
    assert cout % 2 == 0
    x = fcn_block(x, cout // 2, name)
    max = ElementwiseMaxPool(keepdims=True, name=name + '_maxpool')(x)
    max = RepeatElements(x.shape[T_AXIS].value, axis=T_AXIS, name=name + '_repeat')(max)
    x = Concatenate(name=name + '_concat')([max, x])
    return x


def mid_conv_block(x, cout, k, s, p, name='conv'):
    x = ZeroPadding3D(p, name=name + '_pad')(x)
    x = Conv3D(cout, k, strides=s, name=name + '_conv')(x)
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def _rpn_conv(x, cout, k, s, p, name, i=None):
    if i is not None:
        name += '_blk' + str(i)
    x = ZeroPadding2D(p, name=name + '_pad')(x)
    x = Conv2D(cout, k, strides=s, name=name + '_conv')(x)
    x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def rpn_conv_block(x, cout, q, name='rpn_conv'):
    x = _rpn_conv(x, cout, 3, 2, 1, name)
    for i in range(q):
        x = _rpn_conv(x, cout, 3, 1, 1, name, i + 1)
    return x


def make():
    Dp = 10
    Hp = 400
    Wp = 352
    T = 35
    x = Input((Dp, Hp, Wp, T, 7))
    y = vfe_block(x, 32, 'vfe1')
    y = vfe_block(y, 128, 'vfe2')
    y = fcn_block(y, 128, 'fcn1')
    y = ElementwiseMaxPool(name='maxpool')(y)
    y = mid_conv_block(y, 64, 3, (2, 1, 1), (1, 1, 1), name='mid_conv1')
    y = mid_conv_block(y, 64, 3, (1, 1, 1), (0, 1, 1), name='mid_conv2')
    y = mid_conv_block(y, 64, 3, (2, 1, 1), (1, 1, 1), name='mid_conv3')
    # At this point, the output shape is (2, Hp, Wp, 64),
    # because we're using the 'channels_last' data format (it's required for the Dense layers to work)
    #
    # However, the paper is using the 'channels_first' data format, which yields the shape:
    # (64, 2, Hp, Wp) here. It is then reshaped to (128, Hp, Wp).
    #
    # Since the ordering of our dimensions is different from that of the paper, we cannot simply
    # reshape to the desired shape; the spatial relationships of the dimensions should be preserved.
    # Thus, prior to reshaping, we permute the output such that the dimensions to be 'combined'
    # are 'near' each other. That said, we permute such that the shape becomes:
    # (Hp, Wp, 64, 2)
    y = Permute((2, 3, 4, 1))(y)
    # Then reshape it to: (Hp, Wp, 128), consistent with the 'channels_last' data format.
    y = Reshape((Hp, Wp, -1))(y)
    y = rpn_conv_block(y, 128, 3, name='rpn_conv1')
    y_deconv1 = Conv2DTranspose(256, 3, strides=1, padding='same', name='rpn_deconv1')(y)
    y = rpn_conv_block(y, 128, 5, name='rpn_conv2')
    y_deconv2 = Conv2DTranspose(256, 2, strides=2, padding='same', name='rpn_deconv2')(y)
    y = rpn_conv_block(y, 256, 5, name='rpn_conv3')
    y = Conv2DTranspose(256, 4, strides=4, padding='same', name='rpn_deconv3')(y)
    y = Concatenate()([y, y_deconv2, y_deconv1])
    y1 = Conv2D(2, 1, strides=1, padding='same', name='cls')(y)
    y2 = Conv2D(14, 1, strides=1, padding='same', name='reg')(y)
    m = Model(x, [y1, y2])
    return m

import numpy as np
from keras.utils.vis_utils import plot_model


def main():
    model = make()
    plot_model(model, show_shapes=True)
    model.summary()

    model.compile('sgd', ['mse', 'mse'])

    Dp = 10
    Hp = 400
    Wp = 352
    T = 35
    x = np.random.random((1, Dp, Hp, Wp, T, 7))
    y1 = np.random.random((1, Hp//2, Wp//2, 2))
    y2 = np.random.random((1, Hp // 2, Wp // 2, 14))

    def gen():
        yield x, [y1, y2]

    #model.fit_generator(gen(), steps_per_epoch=1, epochs=1)


if __name__ == '__main__':
    main()