from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout

mc = True

def get_dropout(input_tensor, rate, mc=False):
    if mc:
        return Dropout(rate=rate)(input_tensor, training=True)
    else:
        return Dropout(rate=rate)(input_tensor)

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

denseNet = DenseNet169(weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False)
denseNet.trainable = False
denseNet_feature = denseNet(inputs)

conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = MaxPool2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = MaxPool2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = MaxPool2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(conv3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = MaxPool2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(conv4)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = MaxPool2D(pool_size=(2, 2))(conv5)

concatenated_tensor = Concatenate(axis=1)([Flatten()(conv5), Flatten()(denseNet_feature)])

x = Flatten()(concatenated_tensor)

x = Dense(512, activation='relu')(x)

x = get_dropout(x, rate=0.2, mc=mc)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

x = get_dropout(x, rate=0.2, mc=mc)
x = Dense(64, activation='relu')(x)

outputs = Dense(n_classes, activation='sigmoid')(x)

if __name__ == "__main__":
    
    model = Model(inputs=inputs, outputs=outputs) 

    model.summary()