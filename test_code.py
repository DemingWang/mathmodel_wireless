
import tensorflow as tf
# from utils import *
from tensorflow.python.platform import gfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# filename = './test_set/train_117701.csv'
# x_data, y_true = get_batch_data(filename)

# sess = tf.Session()
# pb_file_path = './epoch_49/saved_model.pb'
# sess.run(tf.global_variables_initializer())
# model_f = tf.gfile.FastGFile(pb_file_path, mode='rb')
# graph_def = tf.GraphDef()
# graph_def.ParseFromString(model_f.read())
# ins, outs = tf.import_graph_def(graph_def, return_elements=["inputs:0", "outputs:0"])
# y_pred = sess.run(outs, feed_dict={ins: x_data})

# model = "./epoch_49/saved_model.pb"
# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
# graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
# tf.import_graph_def(graph_def, name='graph')
# summaryWriter = tf.summary.FileWriter('log/', graph)


sess = tf.Session()
print(212313)
tf.saved_model.loader.load(sess, ["serve"], "./model_tf18cpu_1/")
print(323123)
graph = tf.get_default_graph()


print(333)

x = sess.graph.get_tensor_by_name('myInput:0')
y = sess.graph.get_tensor_by_name('myOutput:0')
print(222)
sess.run(tf.global_variables_initializer())


df = pd.read_csv('../autoencoder/test_set/train_278801.csv')

df_np = np.array(df)

train_set = df_np[:,:17]
train_target = df_np[:,17]

train_target = np.expand_dims(train_target, axis=-1)

print(train_set.shape)
print(train_target.shape)
print(train_set[0,:])

mm1 = MinMaxScaler()
mm2 = MinMaxScaler()
train_set = mm1.fit_transform(train_set)
# train_target = mm2.fit_transform(train_target)


print(000)
out = sess.run(y, feed_dict={x:train_set})

# ori_pre,train_target = mm2.inverse_transform(out), mm2.inverse_transform(train_target)
min_rsrp = -150
max_rsrp = -30
ori_pre = out.copy()
ori_pre = min_rsrp + ori_pre * (max_rsrp - min_rsrp)

print(type(ori_pre))
print(ori_pre.shape)
print('RMSE:',np.sqrt(metrics.mean_squared_error(train_target,ori_pre)))
ori_pre = np.squeeze(ori_pre)
train_target = np.squeeze(train_target)
np.savetxt('out.txt',ori_pre)
np.savetxt('y_true.txt',train_target)

print(111)

    
fig = plt.figure()
plt.scatter(train_target,ori_pre)
plt.show()

# t = -103

# tp = len(y_true[(y_true < t) & (out == 1)])
# fp = len(y_true[(y_true >= t) & (out == 1)])
# fn = len(y_true[(y_true < t) & (out == 1)])
# tn = len(y_true[(y_true > t) & (out == 1)])
# print("tp:{},fp:{},fn:{},tn:{}".format(tp, fp, fn, tn))

# precision = tp*1.0/(tp+fp)
# recall = tp*1.0/(tp+fn)
# accuracy = (tp+tn)*1.0/(tp+fn+fp+tn)
# if precision + recall == 0:
#     pcrr = 1
# else:
#     pcrr = 2 * (precision * recall)/(precision + recall)
#     print("precision:{},recall:{},accuracy:{}".format(recision, recall, accuracy))
#     print("pcrr:{}".format(pcrr))
