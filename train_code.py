import tensorflow as tf
import tensorflow.contrib.layers as layers
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt




df = pd.read_csv('./test.csv')

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

min_rsrp = -150
max_rsrp = -30
train_target = np.clip(train_target, min_rsrp, max_rsrp)
train_target = (train_target - min_rsrp)/(max_rsrp - min_rsrp)

# train_target = mm2.fit_transform(train_target)


print(train_set.shape)
print(train_target.shape)

m = len(train_set)
n = 17
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


x = tf.placeholder(tf.float32, shape=[None,n], name="myInput")
y = tf.placeholder(tf.float32, shape=[None,1], name="y")

y_hat = multilayer_perceptron(x)
tf.identity(y_hat, name="myOutput")

correct_prediction = tf.square(y - y_hat)
mse = tf.reduce_mean(tf.cast(correct_prediction,'float'))
train = tf.train.AdamOptimizer(learning_rate=eta).minimize(mse)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(max_epochs):
        _,l,p = sess.run([train, mse, y_hat], feed_dict={x: train_set, y:train_target })

        if i % 100 == 0:
            print('Epoch {}: loss: {}'.format(i, l))
    
    print('train done')

    tf.saved_model.simple_save(sess,'./model_tf18cpu_1/',inputs={'myInput':x}, outputs={'myOutput':y_hat})

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