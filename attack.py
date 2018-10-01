import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

DIVIDER = '---------------------------'
TRAIN = True
DEBUG_PRINTS = True
BATCH_SZ = 128
DEFAULT_EPOCHS = 10
EPS = 0.01
STEPS = 15

class Model:
	def __init__(self):

		self.x = tf.placeholder(tf.float32, shape=(None,28,28))
		self.labels = tf.placeholder(tf.int32, shape=(None,))
		self.x_flat = tf.layers.Flatten()(self.x)
		self.layer1 = tf.layers.dense(self.x_flat, 128, tf.nn.relu)
		self.dropout = tf.layers.dropout(self.layer1, rate=0.20)
		self.logits = tf.layers.dense(self.dropout, 10)
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits))
		self.train_op = tf.train.AdamOptimizer(0.002).minimize(self.loss)
		self.init_op = tf.initializers.global_variables()

		self.saver = tf.train.Saver()


class AdvModel:
	def __init__(self, model, eps=EPS):
		self.x_input = tf.placeholder(tf.float32, (None, 28,28))
		self.x_adv = tf.identity(model.x)
		self.target_class_input = tf.placeholder(tf.int32, shape=(None,))
		self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_class_input,logits=model.logits)
		self.dydx = tf.gradients(self.cross_entropy, model.x)[0]
		self.x_adv = self.x_adv - (eps * self.dydx)
		self.x_adv = tf.clip_by_value(self.x_adv, 0.0, 1.0)

def load_data():
	mnist = keras.datasets.mnist
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	train_images = train_images / 255.0
	test_images = test_images / 255.0

	if DEBUG_PRINTS:
		print('>>>   MNIST Dataset Info   <<<\n' + DIVIDER)
		print('# of training images:\t%d' % (train_images.shape[0]))
		print('Size of images:\t\t(%d, %d)' % (train_images.shape[1], train_images.shape[2]))

	return train_images, train_labels, test_images, test_labels

def train_model(session, model, timgs, tlabels, epochs=DEFAULT_EPOCHS, batch_sz=BATCH_SZ, save=True):
	def shuffle(a, b):
		p = np.random.permutation(len(a))
		return a[p], b[p]

	for i in range(epochs):
		# Shuffle dataset every epoch
		timg_s, tlabels_s = shuffle(train_images, train_labels)
		t = int(np.ceil(len(train_images) * 1.0 / batch_sz))
		total_l = 0
		for j in range(t):
			start = batch_sz * j
			end = min(batch_sz * j + batch_sz, len(timg_s))
			_,l = session.run([model.train_op, model.loss], feed_dict={
													model.x:timg_s[start:end], 
													model.labels:tlabels_s[start:end]
													})
			total_l += l
		print('Total Loss:\t', total_l)

	save_path = model.saver.save(session, "./tmp/model.ckpt")

def load_model(session, model):
	model.saver.restore(session, "./tmp/model.ckpt")
	return

def evaluate_model(session, model, test_imgs, test_labels):
	correct_pred = tf.argmax(model.logits, 1)
	res = session.run([correct_pred], feed_dict={
										model.x:test_imgs, 
										model.labels:test_labels
										})
	count = 0
	for i in range(len(test_labels)):
		if res[0][i] == test_labels[i]:
			count += 1
	if DEBUG_PRINTS:
		print('Accuracy:\t', count / len(test_labels) * 100.0, '%')
	return count / len(test_labels)

def setup_session(model):
	sess = tf.Session()
	sess.run(model.init_op)
	return sess

# Load MNIST data
train_images, train_labels, test_images, test_labels = load_data()
# Create our basic model
model = Model()
# Create TF session
session = setup_session(model)
# Train the model or load existing weights
if not TRAIN:
	load_model(session, model)
else:
	train_model(session, model, train_images, train_labels)
# Evaluate model
evaluate_model(session, model, test_images, test_labels)
# Create FSGM method
fgsm = AdvModel(model)
# Input image
adv_images = [test_images[1]]
# iterate through model
for i in range(STEPS):
	adv_images, ls = session.run([fgsm.x_adv, model.logits],feed_dict={
										model.x: adv_images,
										fgsm.x_input: [test_images[1]],
										fgsm.target_class_input: [5]
	})
	# Printing logits
	if DEBUG_PRINTS:
		print(ls)

adv_img = adv_images[0]
correct_pred = tf.argmax(model.logits, 1)

res, res1 = session.run([model.logits, correct_pred], feed_dict={
										model.x: [adv_img], 
										model.labels: [5]
										})
#print(res)
print('Predicted Class:\t', res1)
plt.imshow(adv_img, cmap=plt.cm.binary)
plt.show()
