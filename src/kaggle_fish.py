# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(2016)
import os
import glob
# 用它可以查找符合特定规则的文件路径名。#获取指定目录下的所有图片  glob.glob(r"E:\Picture\*\*.jpg")
import cv2
# 图像处理的库
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
# 交叉验证的方法
from keras.models import Sequential
# 模型
from keras.layers.core import Dense, Dropout, Flatten
# 常用的网络层 Dense就是常用的全连接层，Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接，
# Dropout层用于防止过拟合。Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# 输出2D图像
from keras.optimizers import SGD
# 优化器
from keras.callbacks import EarlyStopping
# 当监测值不再改善时，该回调函数将中止训练
from keras.utils import np_utils
from sklearn.metrics import log_loss
# log_loss函数，通过给定一列真实值label和一个概率矩阵来计算log loss，返回值通过estimator的predict_proba返回
from keras import __version__ as keras_version
from keras.preprocessing.image import ImageDataGenerator


def get_im_cv2(path):
    # 改变图像的大小
    # 读取并显示图像
    img = cv2.imread(path)
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
    # CV_INTER_NN - 最近邻插值,
    # CV_INTER_LINEAR - 双线性插值(缺省使用)
    # CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法
    # CV_INTER_CUBIC - 立方插值
    return resized


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        # join()   用于将分离的各部分组合成一个路径名
        path = os.path.join('..','kaggle','train',fld,'*.jpg')
        files = glob.glob(path)
        for fl in files:
            # basename()   用于去掉目录的路径，只返回文件名
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            # 图片转化的矩阵
            X_train.append(img)
            # 图片的文件名
            X_train_id.append(flbase)
            # 图片所在label的ID
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('..', 'kaggle', 'test', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    # 矩阵的转置
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255

    # label为0~7共8个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
    # 生成sparse矩阵，存在为1，不存在为0
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def create_model():
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
    # 4个卷积核，每个卷积核大小3*3
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    # 第二个卷积层
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    # 全连接层，先将前一层输出的二维特征图flatten为一维的。
    # Dense就是隐藏层。32就是上一层输出的特征图个数。
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    # 网络到达了分类层。在其之前是一个与类别数相同的Dense层，将特征映射为最终的类别，然后通过一个softmax映射为类别概率。
    # 我们这里是8分类，因此最后的Dense层神经元数是8。这样，整个网络就搭建完毕。
    model.add(Dense(8, activation='softmax'))

    # 开始训练模型，  使用SGD + momentum，  model.compile里的参数loss就是损失函数(目标函数)
    # SGD随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model

def run_cross_validation_create_models(nfolds):
    # input image dimensions
    batch_size = 32
    nb_epoch = 30

    train_data, train_target, train_id = read_and_normalize_train_data()

    # 创造一个字典，方便后期DataFrame
    yfull_train = dict()
    # n_foldfs 默认为10折交叉验证，9/10作为训练集，1/10作为测试集
    # len(train_id) total number of elements
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=None)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf.split(train_data):
        model = create_model()
        # 将train data 划分为训练集和测试集
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        # 当监测值不再改善时，该回调函数将中止训练，monitor：需要监视的量patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
        # verbose：信息展示模式mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]

        # 调用fit方法，就是一个训练过程. 训练的epoch数，batch_size．
        # 数据经过随机打乱shuffle=True。verbose=1，训练过程中输出    的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
        # validation_split=0.2，将20%的数据作为验证集。
        # batch 指的是小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性
        # nb_epoch 整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
        # validation_data 形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt
        print 'Using real-time data augmentation'
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            dim_ordering='th')  # randomly flip images

        datagen.fit(X_train)

        model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),
                            samples_per_epoch=len(X_train),
                            nb_epoch=nb_epoch,
                            callbacks=callbacks,
                            validation_data=(X_valid, Y_valid))

        # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
        #           shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
        #           callbacks=callbacks)

        # 将预测输出，预测测试集中即X_valid的target
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        # 计算log损失，通过已知的target Y_valid和预测的target predictions_valid计算score
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        # 将不同训练的模型添加进models
        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    # tolist()来转换numpy array
    return a.tolist()


def create_submission(predictions, test_id, info):
    # 设定一个列的顺序，DataFrame的列将会精确的按照这个顺序排列
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    # 获取多行数据，并且添加一列image，将其赋值为test_id
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def run_cross_validation_process_test(info_string, models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        # 用不同的model去预测test_data
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        # 相当于把所有的test data都进行了一次prediction
        yfull_test.append(test_prediction)

    # 将每次预测的结果进行算数平均
    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                  + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


print('Keras version: {}'.format(keras_version))
num_folds = 3
info_string, models = run_cross_validation_create_models(num_folds)
run_cross_validation_process_test(info_string, models)
