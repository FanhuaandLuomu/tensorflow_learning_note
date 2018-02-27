#coding:utf-8
# tensorflow学习笔记 day1
# tf入门+线性回归+逻辑回归（mnist-softmax分类器）
from __future__ import division
import tensorflow as tf

# tf.cnstant 创建常量
hello=tf.constant('Hello, Tensorflow')

sess=tf.Session()
print(sess.run(hello))

a=tf.constant(2)
b=tf.constant(3)
with tf.Session() as sess:
	print('a=2,b=3')
	print('a+b=%d' %(sess.run(a+b)))
	print('a-b=%d' %(sess.run(a-b)))
	print('a*b=%d' %(sess.run(a*b)))
	print('a/b=%f' %(sess.run(a/b)))

# tf.placeholder创建变量
a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)
# +
add=tf.add(a,b)
# *
mul=tf.multiply(a,b)
with tf.Session() as sess:
	print('a+b=%d' %(sess.run(add,feed_dict={a:2,b:3})))
	print('a*b=%d' %(sess.run(mul,feed_dict={a:2,b:3})))

# 常量
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])

# 变量
matrix3=tf.placeholder(tf.float32,shape=(1,2))
matrix4=tf.placeholder(tf.float32,shape=(2,1))

print(matrix1.shape)
print(matrix2.shape)
# 矩阵乘法
product=tf.matmul(matrix1,matrix2)
product2=tf.matmul(matrix3,matrix4)
with tf.Session() as sess:
	print(sess.run(product))
	print(sess.run(product2,feed_dict={matrix3:[[4.,4.]],matrix4:[[2.],[2.]]}))

# 线性回归
import numpy
import matplotlib.pyplot as plt
rng=numpy.random

# 参数
lr_rate=0.01
nb_epochs=100
display_step=50

# training data
train_X=numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y=numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_samples=train_X.shape[0]

# tf Graph Input
X=tf.placeholder('float')
Y=tf.placeholder('float')

# Create Model

# Set model weights
# rng.randn() 生成高斯分布 u(0,1) 默认一个数
W=tf.Variable(rng.randn(),name='weight')
b=tf.Variable(rng.randn(),name='bias')

# linear model
output=tf.add(tf.multiply(X,W),b)

# 最小化平方差损失 mse
cost=tf.reduce_sum(tf.pow(output-Y,2))/(2*n_samples)

# sgd
optimizer=tf.train.GradientDescentOptimizer(lr_rate).minimize(cost)

# 初始化参数
init=tf.initialize_all_variables()

with tf.Session() as sess:
	# 初始化权重
	sess.run(init)

	# fit
	for epoch in range(nb_epochs):
		# 每个样本更新一次梯度
		for (x,y) in zip(train_X,train_Y):
			sess.run(optimizer,feed_dict={X:x,Y:y})

		# 打印
		if epoch%display_step==0:
			print('epoch:%04d' %(epoch+1),'cost=%.9f' %(sess.run(cost,feed_dict={X:train_X,Y:train_Y})),\
					'W=',sess.run(W),'b=',sess.run(b))

	print('train Finished')
	print('cost=%.9f' %(sess.run(cost,feed_dict={X:train_X,Y:train_Y})),'W=',sess.run(W),'b=',sess.run(b))

# 逻辑回归 softmax多分类  mnist  epoch-25:0.914
from keras.datasets import mnist
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test)=mnist.load_data()
# reshape 归一化
x_train=x_train.reshape(-1,28*28)/255.0
x_test=x_test.reshape(-1,28*28)/255.0

# 离散化 one-hot
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

# (60000,28,28)  (60000,)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

train_nums=x_train.shape[0]

# 参数
lr_rate=0.01
nb_epochs=25
batch_size=100
display_step=1

# tf Graph Input
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

# model weights
# W=tf.Variable(rng.randn(784,10))
# b=tf.Variable(rng.randn(10))
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

# construct model
pred=tf.nn.softmax(tf.matmul(x,W)+b)  # softmax

# 最小化交叉熵损失
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))

# GD
optimizer=tf.train.GradientDescentOptimizer(lr_rate).minimize(cost)

# 初始化参数
init=tf.initialize_all_variables()

# run Launch the graph
with tf.Session() as sess:
	sess.run(init)

	# training cycle
	for epoch in range(nb_epochs):
		avg_cost=0
		total_batch=int(train_nums/batch_size)
		# loop over all batches
		for i in range(total_batch):
			batch_xs,batch_ys=x_train[batch_size*i:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
			# print(batch_xs.shape,batch_ys.shape)
			_,c=sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
			# print(c)

			avg_cost+=c
		# average loss
		avg_cost=c/total_batch

		if (epoch+1)%display_step==0:
			print('epoch:',epoch+1,'cost=',avg_cost)

	print('Train Finished')
	# Test model
	correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
	# 类型转换+求正确率
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	print('Test shape:',x_test.shape)
	print('Accuracy:',accuracy.eval({x:x_test,y:y_test}))
