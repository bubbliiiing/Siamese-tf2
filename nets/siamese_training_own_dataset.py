import os
import random
from random import shuffle

import cv2
import numpy as np
from PIL import Image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Generator:
    def __init__(self, image_size, dataset_path, batch_size, train_ratio=0.9):
        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]

        self.batch_size = batch_size
        self.train_lines = []
        self.train_labels = []

        self.val_lines = []
        self.val_labels = []
        self.types = 0

        self.train_ratio = train_ratio

        self.load_dataset()

    def load_dataset(self):
        # 遍历dataset文件夹下面的images_background文件夹
        train_path = os.path.join(self.dataset_path, 'images_background')
        for character in os.listdir(train_path):
            # 遍历种类。
            character_path = os.path.join(train_path, character)
            for image in os.listdir(character_path):
                self.train_lines.append(os.path.join(character_path, image))
                self.train_labels.append(self.types)
            self.types += 1

        random.seed(1)
        shuffle_index = np.arange(len(self.train_lines), dtype=np.int32)
        shuffle(shuffle_index)
        random.seed(None)
        self.train_lines = np.array(self.train_lines,dtype=np.object)
        self.train_labels = np.array(self.train_labels)
        self.train_lines = self.train_lines[shuffle_index]
        self.train_labels = self.train_labels[shuffle_index]
        
        num_train = int(len(self.train_lines)*self.train_ratio)

        self.val_lines = self.train_lines[num_train:]
        self.val_labels = self.train_labels[num_train:]
    
        self.train_lines = self.train_lines[:num_train]
        self.train_labels = self.train_labels[:num_train]

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, flip_signal=False):
        if self.channel==1:
            image = image.convert("RGB")

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.75,1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        
        # flip image or not
        flip = rand()<.5
        if flip and flip_signal: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (255,255,255))

        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand()<.5
        if rotate: 
            angle=np.random.randint(-5,5)
            a,b=w/2,h/2
            M=cv2.getRotationMatrix2D((a,b),angle,1)
            image=cv2.warpAffine(np.array(image),M,(w,h),borderValue=[255,255,255]) 

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        
        if self.channel==1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        return image_data

    def _convert_path_list_to_images_and_labels(self, path_list):
        #-------------------------------------------#
        #   如果batch_size = 16
        #   len(path_list)/2 = 32
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros((number_of_pairs, self.image_height, self.image_width, self.channel)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair * 2])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            image = image / 255
            if self.channel == 1:
                pairs_of_images[0][pair, :, :, 0] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image

            image = Image.open(path_list[pair * 2 + 1])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            image = image / 255
            if self.channel == 1:
                pairs_of_images[1][pair, :, :, 0] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image
                
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        # 随机的排列组合
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        return pairs_of_images, labels

    def generate(self, train=True):
        while 1:
            if train:
                lines = self.train_lines
                labels = self.train_labels
            else:
                lines = self.val_lines
                labels = self.val_labels

            batch_images_path = []
            for _ in range(self.batch_size):
                c               = random.randint(0, self.types - 1)
                selected_path   = lines[labels[:] == c]
                while len(selected_path)<3:
                    c               = random.randint(0, self.types - 1)
                    selected_path   = lines[labels[:] == c]
                image_indexes = random.sample(range(0, len(selected_path)), 3)
                # 取出两张类似的图片
                batch_images_path.append(selected_path[image_indexes[0]])
                batch_images_path.append(selected_path[image_indexes[1]])

                # 取出两张不类似的图片
                batch_images_path.append(selected_path[image_indexes[2]])
                # 取出与当前的小类别不同的类
                different_c         = list(range(self.types))
                different_c.pop(c)
                different_c_index   = np.random.choice(range(0, self.types - 1), 1)
                current_c           = different_c[different_c_index[0]]
                selected_path       = lines[labels == current_c]
                while len(selected_path)<1:
                    different_c_index   = np.random.choice(range(0, self.types - 1), 1)
                    current_c           = different_c[different_c_index[0]]
                    selected_path       = lines[labels == current_c]

                image_indexes = random.sample(range(0, len(selected_path)), 1)
                batch_images_path.append(selected_path[image_indexes[0]])

            images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
            yield images, labels
