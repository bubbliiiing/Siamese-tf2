import os

import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.siamese import siamese
from nets.siamese_training import Generator
from nets.siamese_training_own_dataset import \
    Generator as Generator_own_dataset
from utils.utils import ModelCheckpoint


def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'images_background')
        for character in os.listdir(train_path):
            # 在大众类下遍历小种类。
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    else:
        train_path = os.path.join(path, 'images_background')
        for alphabet in os.listdir(train_path):
            # 然后遍历images_background下的每一个文件夹，代表一个大种类
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                # 在大众类下遍历小种类。
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    dataset_path = "datasets"
    #----------------------------------------------------#
    #   训练好的权值保存在logs文件夹里面
    #----------------------------------------------------#
    log_dir = "logs/"
    #----------------------------------------------------#
    #   输入图像的大小，默认为105,105,3
    #----------------------------------------------------#
    input_shape = [105,105,3]
    #----------------------------------------------------#
    #   训练自己的数据的话需要把train_own_data设置成true
    #   训练自己的数据和训练omniglot数据格式不一样
    #----------------------------------------------------#
    train_own_data = False

    model = siamese(input_shape)
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #------------------------------------------------------#
    model_path = 'model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(model_path, by_name=True, skip_mismatch=True)
    
    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    tensorboard = TensorBoard(log_dir=log_dir)
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    train_ratio = 0.9
    images_num = get_image_num(dataset_path, train_own_data)
    train_num = int(images_num*0.9)
    val_num = int(images_num*0.1)
    
    if True:
        Batch_size = 32
        Lr = 1e-3
        Init_epoch = 0
        Freeze_epoch = 50
        
        model.compile(loss = "binary_crossentropy",
                optimizer = Adam(lr=Lr),
                metrics = ["binary_accuracy"])
        print('Train with batch size {}.'.format(Batch_size))

        if train_own_data:
            gen = Generator_own_dataset(input_shape, dataset_path, Batch_size, train_ratio)
        else:
            gen = Generator(input_shape, dataset_path, Batch_size, train_ratio)
            
        model.fit(gen.generate(True),
                steps_per_epoch=max(1,train_num//Batch_size),
                validation_data=gen.generate(True),
                validation_steps=max(1,val_num//Batch_size),
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])
    
    if True:
        Batch_size = 32
        Lr = 1e-4
        Freeze_epoch = 50
        Epoch = 100
        
        model.compile(loss = "binary_crossentropy",
                optimizer = Adam(lr=Lr),
                metrics = ["binary_accuracy"])
        print('Train with batch size {}.'.format(Batch_size))

        if train_own_data:
            gen = Generator_own_dataset(input_shape, dataset_path, Batch_size, train_ratio)
        else:
            gen = Generator(input_shape, dataset_path, Batch_size, train_ratio)
            
        model.fit_generator(gen.generate(True),
                steps_per_epoch=max(1,train_num//Batch_size),
                validation_data=gen.generate(True),
                validation_steps=max(1,val_num//Batch_size),
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard])
