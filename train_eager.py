import time
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from functools import partial
from nets.siamese import siamese
from nets.siamese_training import Generator
from nets.siamese_training_own_dataset import Generator as Generator_own_dataset
from tqdm import tqdm

# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs1, imgs2, targets, net, optimizer):
        with tf.GradientTape() as tape:
            # 计算loss
            prediction = net([imgs1, imgs2], training=True)
            loss_value = tf.reduce_mean(K.binary_crossentropy(targets, prediction))

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        equal = tf.equal(tf.round(prediction),targets)
        accuracy = tf.reduce_mean(tf.cast(equal,tf.float32))
        return loss_value, accuracy
    return train_step

@tf.function
def val_step(imgs1, imgs2, targets, net, optimizer):
    # 计算loss
    prediction = net([imgs1, imgs2], training=False)
    loss_value = tf.reduce_mean(K.binary_crossentropy(targets, prediction))

    return loss_value

def fit_one_epoch(net, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, train_step):
    total_loss = 0
    val_loss = 0
    total_accuracy = 0

    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, targets = batch[0], batch[1]
            images0, images1 = images[0], images[1]
            targets = tf.cast(tf.convert_to_tensor(targets),tf.float32)
            loss_value, accuracy = train_step(images0, images1, targets, net, optimizer)
            total_loss += loss_value.numpy()
            total_accuracy += accuracy.numpy()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'Total Loss'        : total_loss / (iteration + 1), 
                                'Total accuracy'    : total_accuracy / (iteration + 1),
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy(),
                                's/step'            : waste_time})
            pbar.update(1)
            start_time = time.time()
        
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            images, targets = batch[0], batch[1]
            images0, images1 = images[0], images[1]
            targets = tf.convert_to_tensor(targets)

            loss_value = val_step(images0, images1, targets, net, optimizer)
            # 更新验证集loss
            val_loss = val_loss + loss_value.numpy()
            
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'Val Loss'  : val_loss / (iteration + 1), 
                                's/step'    : waste_time})
            pbar.update(1)
            start_time = time.time()

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      

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
    input_shape = [105,105,3]
    dataset_path = "./datasets"
    log_dir = "logs/"
    train_own_data = False

    model = siamese(input_shape)
    model.summary()
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    model_path = 'model_data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    train_ratio = 0.9
    images_num = get_image_num(dataset_path, train_own_data)
    num_train = int(images_num*0.9)
    num_val = int(images_num*0.1)
    
    if True:
        #--------------------------------------------#
        #   Batch_size不要太小，不然训练效果很差
        #--------------------------------------------#
        Batch_size = 32
        Lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 25

        if train_own_data:
            generator = Generator_own_dataset(input_shape, dataset_path, Batch_size, train_ratio)
        else:
            generator = Generator(input_shape, dataset_path, Batch_size, train_ratio)
            
        if Use_Data_Loader:
            gen = partial(generator.generate, train = True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
                
            gen_val = partial(generator.generate, train = False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

        else:
            gen = generator.generate(True)
            gen_val = generator.generate(False)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.95,
            staircase=True
        )

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, get_train_step_fn())


    if True:
        #--------------------------------------------#
        #   Batch_size不要太小，不然训练效果很差
        #--------------------------------------------#
        Batch_size = 32
        Lr = 1e-4
        Freeze_Epoch = 25
        Epoch = 50

        if train_own_data:
            generator = Generator_own_dataset(input_shape, dataset_path, Batch_size, train_ratio)
        else:
            generator = Generator(input_shape, dataset_path, Batch_size, train_ratio)

        if Use_Data_Loader:
            gen = partial(generator.generate, train = True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
                
            gen_val = partial(generator.generate, train = False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

        else:
            gen = generator.generate(True)
            gen_val = generator.generate(False)

        epoch_size = num_train//Batch_size
        epoch_size_val = num_val//Batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.95,
            staircase=True
        )

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, Batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        for epoch in range(Freeze_Epoch,Epoch):
            fit_one_epoch(model, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Epoch, get_train_step_fn())
