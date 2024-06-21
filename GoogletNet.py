import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Input, concatenate, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def inception_module(x, filters, name=None):
    branch1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    branch3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch3x3)

    branch5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch5x5)

    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    if name:
        return concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name=name)
    else:
        return concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)

def GoogLeNet(input_shape=(224, 224, 3), num_classes=1000):
    input = Input(shape=input_shape)
    
    # Initial convolution and pooling
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1_7x7_s2')(input)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1_3x3_s2')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2_3x3_reduce')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_3x3')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool2_3x3_s2')(x)
    
    # Inception modules
    x = inception_module(x, [64, 96, 128, 16, 32, 32], name='mixed3a')
    x = inception_module(x, [128, 128, 192, 32, 96, 64], name='mixed3b')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool3_3x3_s2')(x)
    
    x = inception_module(x, [192, 96, 208, 16, 48, 64], name='mixed4a')
    x = inception_module(x, [160, 112, 224, 24, 64, 64], name='mixed4b')
    x = inception_module(x, [128, 128, 256, 24, 64, 64], name='mixed4c')
    x = inception_module(x, [112, 144, 288, 32, 64, 64], name='mixed4d')
    x = inception_module(x, [256, 160, 320, 32, 128, 128], name='mixed4e')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool4_3x3_s2')(x)
    
    x = inception_module(x, [256, 160, 320, 32, 128, 128], name='mixed5a')
    x = inception_module(x, [384, 192, 384, 48, 128, 128], name='mixed5b')
    
    # Final layers
    x = GlobalAveragePooling2D(name='avgpool')(x)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(1000, activation='relu', name='fc')(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(input, output, name='googlenet')
    return model