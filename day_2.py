#coding:utf-8
# # tensorflow学习笔记 day3
# tf实现多层感知机 MLP  test_acc:0.98+

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import sklearn.preprocessing as prep
import numpy as np
import math

np.random.seed(1234)

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

# (55000,784) (55000,10)
print(mnist.train.images.shape,mnist.train.labels.shape)
# scale:0~1 已归一化 X/255.0
# print(mnist.train.images[0])
# label 已one-hot化
# print(mnist.train.labels[0])

# 标准化处理函数 0均值1方差 (x-mean)/var
def standard_scale(X_train,X_test):
	# 先在train上fit一个模型
	preprocessor=prep.StandardScaler().fit(X_train)
	# train-test 一致
	X_train=preprocessor.transform(X_train)
	X_test=preprocessor.transform(X_test)
	return X_train,X_test


# 标准化
# X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
X_train,X_test=mnist.train.images,mnist.test.images
Y_train,Y_test=mnist.train.labels,mnist.test.labels

print(X_train.shape,X_test.shape)
print(Y_train.shape,Y_test.shape)

# 创建一个Tensorflow默认的 Interactive Session
# sess=tf.InteractiveSession()
sess=tf.Session()

in_units=784
h1_units=300

# 模型参数
# 正态分布 标准差0.1
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1=tf.Variable(tf.zeros([h1_units,]))

W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

# 模型输入
x=tf.placeholder(tf.float32,[None,in_units])
keep_prob=tf.placeholder(tf.float32)

# 定于模型结构
hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)

# 定义loss
# 真实标签
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

# 优化器
train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 训练

# 初始化变量
init=tf.global_variables_initializer()

sess.run(init)


# 1.train on batch  每次训练一个batch
# for i in range(3000):
# 	batch_xs,batch_ys=mnist.train.next_batch(100)
# 	# train_step.run({x:batch_xs,y:batch_ys,keep_prob:0.75})
# 	cost,_=sess.run([cross_entropy,train_step],feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
# 	if i%60==0:
# 		print('nb_batches:',i,'loss:',cost)


# 2.按epoch训练 每次epoch随机打乱
n_samples=len(X_train)  # 55000

dropout=0.75
batch_size=128
nb_epoch=20
for i in range(nb_epoch):
	indices=np.array(range(n_samples))
	np.random.shuffle(indices)
	X_train=X_train[indices]
	Y_train=Y_train[indices]

	avg_loss=0

	total_batches=math.ceil(n_samples/batch_size)
	for bs in range(total_batches):
		batch_xs=X_train[bs*batch_size:(bs+1)*batch_size]
		batch_ys=Y_train[bs*batch_size:(bs+1)*batch_size]

		# print(batch_xs.shape,X_train.shape)

		cost,_=sess.run([cross_entropy,train_step],feed_dict={x:batch_xs,y_:batch_ys,keep_prob:dropout})

		avg_loss+=cost

	print('epoch:%s avg_loss:%s' %(i+1,avg_loss/n_samples))


correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print('train_acc:',accuracy.eval(session=sess,feed_dict={x:mnist.train.images,y_:mnist.train.labels,keep_prob:1.0}))
print('test_acc:',accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
