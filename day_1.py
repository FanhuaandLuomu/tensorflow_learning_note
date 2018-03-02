#coding:utf-8
# # tensorflow学习笔记 day2
# tf实现去噪自编码器  MNIST
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 权重分布 n(0,2/(n_in+n_out))
def xavier_init(n_in,n_out,constant=1):
	low=-constant*np.sqrt(6.0/(n_in+n_out))
	high=constant*np.sqrt(6.0/(n_in+n_out))
	return tf.random_uniform((n_in,n_out),minval=low,maxval=high,dtype=tf.float32)


# 自编码器
class AdditiveGaussianNoiseAutoencoder(object):
	def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
		self.n_input=n_input
		self.n_hidden=n_hidden
		self.transfer=transfer_function
		self.scale=tf.placeholder(tf.float32)
		self.training_scale=scale
		network_weights=self._initialize_weights()
		self.weights=network_weights

		# model
		# 输入
		self.x=tf.placeholder(tf.float32,[None,self.n_input])
		# 隐藏层  x+高斯分布噪声
		self.hidden=self.transfer(tf.add(tf.matmul(\
				self.x+scale*tf.random_normal((n_input,)),\
				self.weights['w1']),self.weights['b1']))
		# 输出
		self.output=tf.add(tf.matmul(self.hidden,\
				self.weights['w2']),self.weights['b2'])

		# loss 平方误差
		self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(\
					self.output,self.x),2.0))

		# 优化
		self.optimizer=optimizer.minimize(self.cost)

		# 初始化
		init=tf.global_variables_initializer()
		self.sess=tf.Session()
		self.sess.run(init)

	# 初始化全部权重
	def _initialize_weights(self):
		all_weights={}
		all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))
		all_weights['b1']=tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
		all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
		all_weights['b2']=tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
		return all_weights

	# 训练一个batch的数据 返回cost
	def partial_fit(self,X):
		# 计算cost+优化
		cost,_=self.sess.run((self.cost,self.optimizer),\
				feed_dict={self.x:X,self.scale:self.training_scale})
		return cost

	# 只计算cost 测试时使用
	def calc_total_cost(self,X):
		cost=self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
		return cost

	# 提取隐藏层的高阶特征
	def transform(self,X):  
		return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})

	# 将高阶特征还原
	def generate(self,hidden=None):
		if hidden is None:
			hidden=np.random.normal(size=(self.n_hidden,))
		return self.sess.run(self.output,feed_dict={self.hidden:hidden})

	# 整个计算图执行一遍 
	def reconstruct(self,X):
		return self.sess.run(self.output,feed_dict={self.x:X,self.scale:self.training_scale})

	# 返回输入层->隐藏层的w
	def getWeights(self):
		return self.sess.run(self.weights['w1'])

	# 返回b
	def getBiases(self):
		return self.sess.run(self.weights['b1'])


mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
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
X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)

n_samples=X_train.shape[0]
print(n_samples)
training_epochs=20
batch_size=128
display_step=1

autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=X_train.shape[1],\
						n_hidden=200,transfer_function=tf.nn.softplus,\
						optimizer=tf.train.AdamOptimizer(learning_rate=0.001),\
						scale=0.01)

import copy
# 深度copy  X_train每次epoch中随机打乱  保留一个原始顺序的副本
X_train_copy=copy.deepcopy(X_train)
# 训练
for epoch in range(training_epochs):

	# 每一次迭代打乱顺序
	indces=np.array(range(n_samples))
	np.random.shuffle(indces)
	X_train=X_train[indces]

	avg_cost=0
	total_batch=int(n_samples/batch_size)
	for i in range(total_batch):
		# 一个batch数据
		batch_xs=X_train[batch_size*i:batch_size*(i+1)]
		# 训练
		cost=autoencoder.partial_fit(batch_xs)

		# 求每个样本的平均误差
		avg_cost+=cost

	avg_cost/=n_samples

	if epoch%display_step==0:
		print('epoch:',epoch+1,'loss:',avg_cost)

# 测试
print('Train AVG Loss:',autoencoder.calc_total_cost(X_train_copy)/len(X_train_copy))
print('Test AVG Loss:',autoencoder.calc_total_cost(X_test)/len(X_test))

# 提取train和test的隐层高阶特征 （相当于特征特征降维、主成分分析PCA）
train_hidden_feats=autoencoder.transform(X_train_copy)
print('the hidden (higher) features of x_train:')
print(train_hidden_feats.shape)
print(train_hidden_feats)

test_hidden_feats=autoencoder.transform(X_test)
print('the hidden (higher) features of x_test:')
print(test_hidden_feats.shape)
print(test_hidden_feats)