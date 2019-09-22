# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.platform import gfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def CaculatePcrr(y_true,y_pred):
    t = -103

    # tp = len(y_true[(y_true < t) & (y_pred < t)])
    # fp = len(y_true[(y_true >= t) & (y_pred < t)])
    # fn = len(y_true[(y_true < t) & (y_pred > t)])
    # tn = len(y_true[(y_true > t) & (y_pred > t)])

    tp = len(y_true[(y_true < t) & (y_pred < t)])
    fp = len(y_true[(y_true >= t) & (y_pred < t)])
    fn = len(y_true[(y_true < t) & (y_pred > t)])
    tn = len(y_true[(y_true > t) & (y_pred > t)])

    print("tp:{},fp:{},fn:{},tn:{}".format(tp, fp, fn, tn))

    precision = tp*1.0/(tp+fp)
    recall = tp*1.0/(tp+fn)
    accuracy = (tp+tn)*1.0/(tp+fn+fp+tn)
    if precision + recall == 0:
        pcrr = 1
    else:
        pcrr = 2 * (precision * recall)/(precision + recall)
        print("precision:{},recall:{},accuracy:{}".format(precision, recall, accuracy))
        print("pcrr:{}".format(pcrr))


def preProcessDataForTesting():
    df = pd.read_csv('../autoencoder/test_set/train_278801.csv')
    df_np = np.array(df, dtype=np.float32)
    train_set = df_np[:,:17]
    train_target = df_np[:,17]

    train_target = np.expand_dims(train_target, axis=-1)

    print(train_set.shape)
    print(train_target.shape)
    print(train_set[0,:])

    mm1 = MinMaxScaler()
    train_set = mm1.fit_transform(train_set)

    return train_set,train_target

train_set,train_target = preProcessDataForTesting()

sess = tf.Session()
tf.saved_model.loader.load(sess, ["serve"], "./model_upload/model1/")
graph = tf.get_default_graph()


x = sess.graph.get_tensor_by_name('myInput:0')
y = sess.graph.get_tensor_by_name('myOutput:0')
sess.run(tf.global_variables_initializer())

out = sess.run(y, feed_dict={x:train_set})

# ori_pre,train_target = mm2.inverse_transform(out), mm2.inverse_transform(train_target)
min_rsrp = -150
max_rsrp = -30
ori_pre = out.copy()
ori_pre = min_rsrp + ori_pre * (max_rsrp - min_rsrp)


print(ori_pre.shape)
print('RMSE:',np.sqrt(metrics.mean_squared_error(train_target,ori_pre)))

ori_pre = np.squeeze(ori_pre)
train_target = np.squeeze(train_target)
np.savetxt('out.txt',ori_pre)
np.savetxt('y_true.txt',train_target)


fig = plt.figure()
plt.scatter(train_target,ori_pre)
#设置坐标轴范围
plt.xlim((-140, -70))
plt.ylim((-140, -70))
plt.show()


#设置坐标轴名称
# plt.xlabel('xxxxxxxxxxx')
# plt.ylabel('yyyyyyyyyyy')
#设置坐标轴刻度
# my_x_ticks = np.arange(-5, 5, 0.5)
# my_y_ticks = np.arange(-2, 2, 0.3)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)


CaculatePcrr(train_target,ori_pre)

