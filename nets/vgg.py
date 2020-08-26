import os
import numpy as np
from tensorflow.keras.layers import Input,Dense,Conv2D
from tensorflow.keras.layers import MaxPooling2D,Flatten,Lambda
from tensorflow.keras.models import Model
from PIL import Image

class VGG16:
    def __init__(self):
        self.block1_conv1 = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')
        self.block1_conv2 = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')
        self.block1_pool = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')
        
        self.block2_conv1 = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')
        self.block2_conv2 = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')
        self.block2_pool = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')

        self.block3_conv1 = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')
        self.block3_conv2 = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')
        self.block3_conv3 = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')
        self.block3_pool = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')

        self.block4_conv1 = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')
        self.block4_conv2 = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')
        self.block4_conv3 = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')
        self.block4_pool = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')

        # 第五个卷积部分
        self.block5_conv1 = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')
        self.block5_conv2 = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')
        self.block5_conv3 = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')   
        self.block5_pool = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')

        self.flatten = Flatten(name = 'flatten')

    def call(self, inputs):
        x = inputs
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pool(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pool(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pool(x)
        
        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_pool(x)

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_pool(x)

        outputs = self.flatten(x)
        return outputs
