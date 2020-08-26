import os
import numpy as np
import tensorflow.keras.backend as K
from PIL import Image
from nets.vgg import VGG16
from tensorflow.keras.layers import Input,Dense,Conv2D
from tensorflow.keras.layers import MaxPooling2D,Flatten,Lambda
from tensorflow.keras.models import Model
 
def siamese(input_shape):
    vgg_model = VGG16()

    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    encoded_image_1 = vgg_model.call(input_image_1)
    encoded_image_2 = vgg_model.call(input_image_2)

    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

    out = Dense(512,activation='relu')(l1_distance)
    out = Dense(1,activation='sigmoid')(out)

    model = Model([input_image_1,input_image_2],out)
    return model
