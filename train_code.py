# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.layers as layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import os



# def preProcessDataForTraining():
#     #导入训练数据集
#     total_train_set_data = []
#     for train_set_file_name in os.listdir("train_set"):
#         with open(os.path.join('train_set',train_set_file_name), "r") as rf:
#             file_data = pd.read_csv(rf)
#             file_data = np.array(file_data.get_values(), dtype=np.float32)
#             print("fileName:", train_set_file_name, "  shape of file data:", file_data.shape)
#             total_train_set_data.extend(file_data)
#     total_train_set_data = np.array(total_train_set_data)
#     print("shape of total_train_set_data data:", total_train_set_data.shape)
#     #至此，我们把“train_set”文件夹下的数据都读到total_train_set_data中了
#     #此时的total_train_set_data是个维度为[15284, 18]的数组
#     #开始对每个数据样本构造数据特征
#     total_train_xs = []
#     total_train_ys = []
#     for i in range(len(total_train_set_data)):
#         org_data = total_train_set_data[i]
#         #假设我们觉得数据中的四个地理坐标可以构造一个距离特征
#         feature_dis_2d = np.sqrt(np.power(org_data[12]-org_data[1],2)+np.power(org_data[13]-org_data[2],2))     
#         #假设我们觉得RS Power这个特征也很重要
#         feature_RS_Power = org_data[8]
#         #假设我们觉得三维空间中的这个距离也很重要
#         feature_dis_3d = np.sqrt(np.power(org_data[12]-org_data[1],2)
#                                  +np.power(org_data[13]-org_data[2],2)
#                                  +np.power(org_data[14]-org_data[9],2))
#         #现在我们构造三个特征了
#         tmp_train_xs = [feature_dis_2d/100, feature_RS_Power, feature_dis_3d/100]
#         #我们期望预测的就是第18列的RSRP
#         tmp_train_ys = [org_data[17]]
#         total_train_xs.append(tmp_train_xs)
#         total_train_ys.append(tmp_train_ys)
#     #将total_train_xs、total_train_ys转换为numpy数组类型
#     total_train_xs = np.array(total_train_xs)
#     total_train_ys = np.array(total_train_ys)    
#     print("shape of total_train_xs:", total_train_xs.shape)    
#     print("shape of total_train_ys:", total_train_ys.shape)
    
#     return total_train_xs, total_train_ys

def get_data_from_file():
    df = pd.read_csv('../Dataset/test.csv')
    df_np = np.array(df, dtype=np.float32)

    train_set = df_np[:,:17]
    train_target = df_np[:,17]

    train_target = np.expand_dims(train_target, axis=-1)

    print(train_set.shape)
    print(train_target.shape)
    print(train_set[0,:])

    mm1 = MinMaxScaler()
    mm2 = MinMaxScaler()
    train_set = mm1.fit_transform(train_set)

    min_rsrp = -150
    max_rsrp = -30

    train_target = np.clip(train_target, min_rsrp, max_rsrp)
    train_target = (train_target - min_rsrp)/(max_rsrp - min_rsrp)

    # train_target = mm2.fit_transform(train_target)
    print(train_set.shape)
    print(train_target.shape)

    return train_set, train_target

NUMOFINPUTS = 17
NUMOFOUTPUTS = 1
n_hidden = 20

batch_size = 1000
eta = 0.001
max_epochs = 5000



def multilayer_perceptron(x):
    x1 = x[:,1:17]
    fc1 = layers.fully_connected(x1,50,activation_fn = tf.nn.relu, scope = "fc1")
    fc2 = layers.fully_connected(fc1,20,activation_fn = tf.nn.relu, scope = "fc2")
    out = layers.fully_connected(fc1,1,activation_fn = tf.sigmoid, scope = "out")
    return out


x = tf.placeholder(tf.float32, shape=[None,NUMOFINPUTS], name="myInput")
y = tf.placeholder(tf.float32, shape=[None,NUMOFOUTPUTS], name="y")

y_hat = multilayer_perceptron(x)
tf.identity(y_hat, name="myOutput")

correct_prediction = tf.square(y - y_hat)
mse = tf.reduce_mean(tf.cast(correct_prediction,'float'))
train = tf.train.AdamOptimizer(learning_rate=eta).minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #导入预处理之后的训练数据
    total_train_xs, total_train_ys = get_data_from_file()

    for i in range(max_epochs):
        #在total_train_xs, total_train_ys数据集中随机抽取batch_size个样本出来
        #作为本轮迭代的训练数据batch_xs, batch_ys
        batch_size = 1000
        num_of_batches = len(total_train_xs) // batch_size
        sample_idxs = range(len(total_train_xs))
        random.shuffle(sample_idxs)
        for j in range(0,num_of_batches):
            batch_xs = []
            batch_ys = []
            sample_idx = sample_idxs[batch_size * j : batch_size *(j+1)]
            for idx in sample_idx:
                batch_xs.append(total_train_xs[idx])
                batch_ys.append(total_train_ys[idx])
            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)            
            #喂训练数据进去训练
            _,l,p = sess.run([train, mse, y_hat], feed_dict={x: batch_xs, y:batch_ys})

        if i % 1 == 0:
            print('Epoch {}: loss: {}'.format(i, l))
    
    print('train done')

    tf.saved_model.simple_save(sess,'./model_upload/model1',inputs={'myInput':x}, outputs={'myOutput':y_hat})

# eval
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    # print('mean error: ', accuracy.eval({x: train_set, y:train_target}))
    # fig = plt.figure()
    # ori_target,ori_pre = mm2.inverse_transform(train_target), mm2.inverse_transform(p)
    # print([ori_target,ori_pre])
    # print('RMSE:',np.sqrt(metrics.mean_squared_error(ori_target,ori_pre)))
    # plt.scatter(ori_target,ori_pre)
    # plt.show()

print('optimize finish')