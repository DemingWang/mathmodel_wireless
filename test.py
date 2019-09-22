# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import copy
import numpy as np

input_size = 100
file_content = '/home/nao/Desktop/train_set/train_115601.csv'
pb_data = pd.read_csv(file_content)
batch_data = np.array(pb_data.get_values()[:,0:18], dtype=np.float32)

while(1):
    if batch_data.shape[0] < input_size*input_size :
        batch_data = np.concatenate((batch_data, batch_data),axis=0)
    else:
        break

batch_data = batch_data[0:input_size*input_size, :]

x_test = batch_data[:,0:17]
x_test = [x_test]
x_test = np.array(x_test, dtype=np.float32)
y_true = batch_data[:,17]
y_true = y_true[:,np.newaxis]
y_true = np.array(y_true, dtype=np.float32)


sess = tf.Session()

tf.saved_model.loader.load(sess, ["serve"], "/home/nao/Desktop/model_new5/epoch_64")

graph = tf.get_default_graph()


inputs_ = sess.graph.get_tensor_by_name('myInput:0')
RSRP = sess.graph.get_tensor_by_name('RSRP:0')
sess.run(tf.global_variables_initializer())

y_pred = sess.run(RSRP, feed_dict = {inputs_ : x_test})
print(y_true.shape)
print(y_pred.shape)
print(y_pred)


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