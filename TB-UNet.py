from keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from parameters import *

def conv_bn_act(x, filters, drop_out=0.0):
    x = Conv2D(filters, (3, 3), activation=None, padding='same')(x)

    if drop_out > 0:
        x = Dropout(drop_out)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def expend_as(x, n):
    y = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': n})(x)
    return y

def attention_layer(d, e, n):
    d1 = Conv2D(n, (1, 1), activation=None, padding='same')(d)
    e1 = Conv2D(n, (1, 1), activation=None, padding='same')(e)

    concat_de = add([d1, e1])

    relu_de = Activation('relu')(concat_de)
    conv_de = Conv2D(1, (1, 1), padding='same')(relu_de)
    sigmoid_de = Activation('sigmoid')(conv_de)

    shape_e = K.int_shape(e)
    upsample_psi = expend_as(sigmoid_de, shape_e[3])

    return multiply([upsample_psi, e])

def df_block(x, filters, compression=0.5, drop_out=0.0):
    x1 = Conv2D(filters, (3, 3), dilation_rate=2, padding='same')(x)

    if drop_out > 0:
        x1 = Dropout(drop_out)(x1)

    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = Conv2D(filters, (3, 3), padding='same')(x)

    if drop_out > 0:
        x2 = Dropout(drop_out)(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = add([x1, x2])

    x3 = GlobalAveragePooling2D()(x3)

    x3 = Dense(int(filters * compression))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x3 = Dense(filters)(x3)

    x3p = Activation('sigmoid')(x3)

    x3m = Lambda(lambda x: 1 - x)(x3p)

    x4 = multiply([x1, x3p])
    x5 = multiply([x2, x3m])

    return add([x4, x5])

def TB_DenseNet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), filters=16, compression=0.5, drop_out=0, half_net=False, attention_gates=True):

    inputShape = Input(input_shape)

    c1 = df_block(inputShape, filters, compression=compression, drop_out=drop_out)
    c1 = df_block(c1, filters, compression=compression, drop_out=drop_out)
    p1 = MaxPooling2D((2, 2))(c1)
    filters = 2 * filters

    c2 = df_block(p1, filters, compression=compression, drop_out=drop_out)
    c2 = df_block(c2, filters, compression=compression, drop_out=drop_out)
    p2 = MaxPooling2D((2, 2))(c2)
    filters = 2 * filters

    c3 = df_block(p2, filters, compression=compression, drop_out=drop_out)
    c3 = df_block(c3, filters, compression=compression, drop_out=drop_out)
    p3 = MaxPooling2D((2, 2))(c3)
    filters = 2 * filters

    c4 = df_block(p3, filters, compression=compression, drop_out=drop_out)
    c4 = df_block(c4, filters, compression=compression, drop_out=drop_out)
    p4 = MaxPooling2D((2, 2))(c4)
    filters = 2 * filters

    cm = df_block(p4, filters, compression=compression, drop_out=drop_out)
    cm = df_block(cm, filters, compression=compression, drop_out=drop_out)

    filters = filters // 2

    u4 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(cm)

    if attention_gates:
        u4 = concatenate([u4, attention_layer(u4, c4, 1)], axis=3)
    else:
        u4 = concatenate([u4, c4], axis=3)

    if half_net:
        c5 = conv_bn_act(u4, filters, drop_out=drop_out)
        c5 = conv_bn_act(c5, filters, drop_out=drop_out)
    else:
        c5 = df_block(u4, filters, compression=compression, drop_out=drop_out)
        c5 = df_block(c5, filters, compression=compression, drop_out=drop_out)

    filters = filters // 2

    u3 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c5)

    if attention_gates:
        u3 = concatenate([u3, attention_layer(u3, c3, 1)], axis=3)
    else:
        u3 = concatenate([u3, c3], axis=3)

    if half_net:
        c6 = conv_bn_act(u3, filters, drop_out=drop_out)
        c6 = conv_bn_act(c6, filters, drop_out=drop_out)
    else:
        c6 = df_block(u3, filters, compression=compression, drop_out=drop_out)
        c6 = df_block(c6, filters, compression=compression, drop_out=drop_out)

    filters = filters // 2

    u2 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c6)

    if attention_gates:
        u2 = concatenate([u2, attention_layer(u2, c2, 1)], axis=3)
    else:
        u2 = concatenate([u2, c2], axis=3)

    if half_net:
        c7 = conv_bn_act(u2, filters, drop_out=drop_out)
        c7 = conv_bn_act(c7, filters, drop_out=drop_out)

    else:
        c7 = df_block(u2, filters, compression=compression, drop_out=drop_out)
        c7 = df_block(c7, filters, compression=compression, drop_out=drop_out)

    filters = filters // 2

    u1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(c7)

    if attention_gates:
        u1 = concatenate([u1, attention_layer(u1, c1, 1)], axis=3)
    else:
        u1 = concatenate([u1, c1], axis=3)

    if half_net:
        c8 = conv_bn_act(u1, filters, drop_out=drop_out)
        c8 = conv_bn_act(c8, filters, drop_out=drop_out)
    else:
        c8 = df_block(u1, filters, compression=compression, drop_out=drop_out)
        c8 = df_block(c8, filters, compression=compression, drop_out=drop_out)

    c9 = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c8)

    return Model(inputs=[inputShape], outputs=[c9])

if __name__ == "__main__":
  
  model = TB_DenseNet(attention_gates=attention_gates)
  
  model.summary()