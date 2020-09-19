from random import shuffle
from PIL import Image
import numpy as np
import os
import random
import cv2
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class Generator:
    def __init__(self, image_size, dataset_path, batch_size, train_ratio=0.9):
        self.dataset_path = dataset_path
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.channel = image_size[2]

        self.batch_size = batch_size
        
        self.train_dictionary = {}
        self._train_chapters = []
        self._validation_chapters = []

        self._current_train_alphabet_index = 0
        self._current_val_alphabet_index = 0

        self.train_ratio = train_ratio

        self.load_dataset()
        self.split_train_datasets()

    def load_dataset(self):
        # 遍历dataset文件夹下面的images_background文件夹
        train_path = os.path.join(self.dataset_path, 'images_background')
        for character in os.listdir(train_path):
            # 遍历种类。
            character_path = os.path.join(train_path, character)
            self.train_dictionary[character] = os.listdir(character_path)

    def split_train_datasets(self):
        available_chapters = list(self.train_dictionary.keys())
        number_of_chapters = len(available_chapters)
        # 进行验证集和训练集的划分
        self._train_chapters = available_chapters[:int(self.train_ratio*number_of_chapters)]

        self._validation_chapters = available_chapters[int(self.train_ratio*number_of_chapters):]

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
        # cv2.imshow("TEST",np.uint8(image_data))
        # cv2.waitKey(0)
        return image_data

    def _convert_path_list_to_images_and_labels(self, path_list):
        # 如果batch_size = 16
        # len(path_list)/ = 32
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros((number_of_pairs, self.image_height, self.image_width, self.channel)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            image = Image.open(path_list[pair * 2])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            # cv2.imwrite("img/"+str(pair)+"_0"+".jpg",np.uint8(image),)
            image = image / 255
            if self.channel == 1:
                pairs_of_images[0][pair, :, :, 0] = image
            else:
                pairs_of_images[0][pair, :, :, :] = image

            image = Image.open(path_list[pair * 2 + 1])
            image = self.get_random_data(image, [self.image_height, self.image_width])
            image = np.asarray(image).astype(np.float64)
            # cv2.imwrite("img/"+str(pair)+"_1"+".jpg",np.uint8(image),)
            image = image / 255
            if self.channel == 1:
                pairs_of_images[1][pair, :, :, 0] = image
            else:
                pairs_of_images[1][pair, :, :, :] = image
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        # print(labels)

        # 随机的排列组合
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        # print(path_list[number_of_pairs],labels)
        return pairs_of_images, labels

    def generate(self,train=True):
        # self._train_alphabets里面存放的是大类别
        if train:
            available_characters = self._train_chapters
        else:
            available_characters = self._train_chapters

        while 1:
            number_of_characters = len(available_characters)

            batch_images_path = []

            # 在小类别里面筛选
            selected_characters_indexes = [random.randint(0, number_of_characters-1) for i in range(self.batch_size)]
            
            # 对于每一个筛选到的小类别
            for index in selected_characters_indexes:
                # 除去小类别的名称
                current_character = available_characters[index]
                # 获取当前这个小类别的路径
                image_path = os.path.join(self.dataset_path, 'images_background', current_character)

                available_images = os.listdir(image_path)
                # print(len(available_images))
                image_indexes = np.random.choice(range(0, len(available_images)), 3)
                # 取出两张类似的图片
                image = os.path.join(image_path, available_images[image_indexes[0]])
                batch_images_path.append(image)
                image = os.path.join(image_path, available_images[image_indexes[1]])
                batch_images_path.append(image)

                # 取出两张不类似的图片
                image = os.path.join(image_path, available_images[image_indexes[2]])
                batch_images_path.append(image)
                # 取出与当前的小类别不同的类
                different_characters = available_characters[:]
                different_characters.pop(index)
                different_character_index = np.random.choice(range(0, number_of_characters - 1), 1)
                current_character = different_characters[different_character_index[0]]
                image_path = os.path.join(self.dataset_path, 'images_background', current_character)

                available_images = os.listdir(image_path)
                image_indexes = np.random.choice(range(0, len(available_images)), 1)
                image = os.path.join(image_path, available_images[image_indexes[0]])
                batch_images_path.append(image)

            images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
            yield images, labels
