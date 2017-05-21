from datetime import datetime
import tensorflow as tf
import numpy as np

server_target="grpc://172.16.100.2:12222"  #master
# create data

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
starttime=datetime.now()
with tf.device("/job:canshu/task:0"):
    Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

with tf.device("/job:canshu/task:1"):
    biases = tf.Variable(tf.zeros([1]))

with tf.device("/job:gongzuo/ps:0"):
    y = Weights*x_data + biases
    loss = tf.reduce_mean(tf.square(y-y_data))

with tf.device("/job:gongzuo/ps:1"):
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

#local-----------------------------------
# Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases = tf.Variable(tf.zeros([1]))
# y = Weights*x_data + biases
# loss = tf.reduce_mean(tf.square(y-y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
# init = tf.global_variables_initializer()
# sess = tf.Session()
#local end--------------------------------

### create tensorflow structure end ###
sess=tf.Session(server_target)
sess.run(init)

for step in range(100):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

endtime=datetime.now()
print((endtime-starttime).seconds)
