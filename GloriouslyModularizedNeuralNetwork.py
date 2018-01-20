# Author: Chao Zhang, Yiting Li

from __future__ import print_function
import datetime
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys

_python_version = sys.version_info[0]
timestamp = "_" + ''.join(c for c in datetime.datetime.now().isoformat() if c.isdigit())

class FullyConnectedLayer:
	def __init__(self, inputSize, outputSize, weightInitialFactor = 1):
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.weight = (np.random.rand(inputSize + 1, outputSize) * 2 - 1) * weightInitialFactor
		self.weightGradient = np.zeros((inputSize + 1, outputSize))
	
	def forward(self, input):
		self.input = np.concatenate((input, np.ones((input.shape[0], 1))), axis=1)
		self.output = np.dot(self.input, self.weight)
		return self.output
	
	def backward(self, gradient):
		self.weightGradient = np.dot(self.input.T, gradient)
		return np.dot(gradient, self.weight.T)[:,:-1]
		
class SigmoidLayer:
	def __init__(self):
		self.type = "sigmoid"
	
	def __eq__(self, other): 
		return self.type == other.type
	
	def forward(self, input):
		absInput = np.absolute(input)
		positiveInput = (input + absInput) / 2
		negativeInput = input - positiveInput
		self.output = 1 / (1 + np.exp(-positiveInput)) + np.exp(negativeInput) / (1 + np.exp(negativeInput)) - 0.5
		return self.output
	
	def backward(self, gradient):
		return gradient * self.output * (1 - self.output)

class SoftmaxLayer:
	def __init__(self):
		self.type = "softmax"
	
	def __eq__(self, other): 
		return self.type == other.type
		
	def forward(self, input):
		temp = np.exp(input)
		tempsum = np.sum(temp, axis=1)
		tempsum = np.reshape(tempsum, (input.shape[0], 1))
		tempsum = np.repeat(tempsum, input.shape[1], axis=1)
		self.output = temp / tempsum
		return self.output
	
	def backward(self, gradient):
		tempsum = np.sum(gradient * self.output, axis=1)
		tempsum = np.reshape(tempsum, (gradient.shape[0], 1))
		tempsum = np.repeat(tempsum, gradient.shape[1], axis=1)
		return self.output * (gradient - tempsum)

class BinaryCrossEntropyLossFunction:
	def loss(self, output, teacher):
		return -teacher * np.log(output + 1e-100) - (1 - teacher) * np.log(1 - output + 1e-100)
		
	def gradient(self, output, teacher):
		return (1 - teacher) / (1 - output + 1e-100) - teacher / (output + 1e-100)

class MultiwayCrossEntropyLossFunction:
	def loss(self, output, teacher):
		return - np.sum(teacher * np.log(output + 1e-100), axis=1)
		
	def gradient(self, output, teacher):
		return - teacher / (output + 1e-100)

class BinaryPredictor:
	def predict(self, output):
		return np.round(output)

class ContinuousPredictor:
	def predict(self, output):
		return output
	
class MaxPredictor:
	def predict(self, output):
		maxi = np.argmax(output, axis=1)
		prediction = np.zeros(output.shape)
		prediction[np.arange(output.shape[0]), maxi] = 1
		return prediction

class NaiveTrainer:
	def __init__(self, stepSize = 0.05):
		self.stepSize = stepSize
	
	def train(self, layer, regularizer):
		if hasattr(layer, 'weight'):
			layer.weight -= regularizer.regularize(layer) * self.stepSize

class AnnealingTrainer:
	def __init__(self, stepSize = 0.05, T = 0.1):
		self.nstep = 0
		self.stepSize = stepSize
		self.T = T
	
	def train(self, layer, regularizer):
		if hasattr(layer, 'weight'):
			layer.weight -= regularizer.regularize(layer) * (self.stepSize / (1 + np.power(self.nstep, 0.5) / self.T))

class MomentumTrainer:
	def __init__(self, stepSize = 0.05, momentumFactor = 0.9):
		self.stepSize = stepSize
		self.momentumFactor = momentumFactor
	
	def train(self, layer, regularizer):
		if hasattr(layer, 'weight'):
			if hasattr(layer, 'momentum'):
				layer.momentum = layer.momentum * self.momentumFactor + regularizer.regularize(layer) * self.stepSize * (1 - self.momentumFactor)
			else:
				layer.momentum = regularizer.regularize(layer) * self.stepSize * (1 - self.momentumFactor)
			layer.weight -= layer.momentum

class NoRegularizer:
	def regularize(self, layer):
		return layer.weightGradient

class L2Regularizer:
	def __init__(self, modifier=1.):
		self.modifier = modifier
	
	def regularize(self, layer):
		return layer.weightGradient + 2 * self.modifier * layer.weight

class L1Regularizer:
	def __init__(self, modifier=1.):
		self.modifier = modifier
	
	def regularize(self, layer):
		return layer.weightGradient + self.modifier * np.sign(layer.weight)
		
class NN:	
	def __init__(self, lossfunction, trainer = NaiveTrainer(), regularizer = NoRegularizer(), predictor = ContinuousPredictor()):
		self.flow = []
		self.lossfunction = lossfunction
		self.trainer = trainer
		self.predictor = predictor
		self.regularizer = regularizer
	
	def addLinearLayers(self, layerList):
		for i in range(len(layerList)):
			if i == len(layerList) - 1:
				self.flow.append((layerList[i], []));
			else:
				self.flow.append((layerList[i], [layerList[i + 1]]));
	
	def test(self, input):
		for step in self.flow:
			layer = step[0]
			input = layer.forward(input)
			#Assume to be linear
		return input
	
	def loss(self, input, teacher):
		return np.mean(self.lossfunction.loss(self.test(input), teacher))
	
	def predict(self, input):
		return self.predictor.predict(self.test(input))
	
	def predictionAccuracy(self, input, teacher):
		return np.sum(np.sum(np.abs(teacher != self.predict(input)), axis=1) == 0) / input.shape[0]
	
	def train(self, input, teacher):
		output = self.test(input)
		gradient = self.lossfunction.gradient(output, teacher)
		for step in reversed(self.flow):
			layer = step[0]
			#Assume to be linear
			gradient = layer.backward(gradient)
			self.trainer.train(layer, self.regularizer)
	
	def getWeights(self):
		weights = []
		for step in self.flow:
			layer = step[0]
			if hasattr(layer, 'weight'):
				weights.append(np.copy(layer.weight))
		return weights
	
	def setWeights(self, weights):
		i = 0
		for step in self.flow:
			layer = step[0]
			if hasattr(layer, 'weight'):
				layer.weight = weights[i]
				i += 1

#											 ::
#											:;J7, :,                        ::;7:
#											,ivYi, ,                       ;LLLFS:
#											:iv7Yi                       :7ri;j5PL
#										 ,:ivYLvr                    ,ivrrirrY2X,
#										 :;r@Wwz.7r:                :ivu@kexianli.
#										:iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
#									 ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
#								;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
#							:,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
#								 ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
#						 ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
#					 :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
#					,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
#				,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
#				:      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
#			 ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
#			,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
#		 ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
#		:,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
#		::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
#		i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
#		:      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
#		i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
#		:    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
#		:     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
#		 :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
#					, :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
#							:jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
#					 ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
#					 ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
#				, , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
#				 :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii:::

def readData(label_fl, image_file, training):
	
	'''
	self function is used to read in data from MNIST data set
	'''
	
	val = 20000 if training else 2000
	label_file =  open(label_fl, 'rb')
	tup = struct.unpack(">II", label_file.read(8))
	labels = np.fromfile(label_file, dtype=np.int8)
	
	img_file = open(image_file, 'rb')
	tup = struct.unpack(">IIII", img_file.read(16))
	images = np.fromfile(img_file, dtype=np.uint8).reshape(len(labels), tup[2], tup[3])
	
	subset = np.zeros((val,tup[2]*tup[3]),dtype=np.float64)
	if training:
		for img in range(20000):
			subset[img] = images[img].flatten()/float(100)
		return (labels[:20000],subset)
	else:
		for img in range(2000):
			subset[img] = images[len(images) - 2000 + img].flatten()/float(100)
		return (labels[len(images) - 2000:],subset)

"""
Parameter:
mode should be "2v3", "2v8", "all"
"""
def preprocessData(mode="2v3", holdoutFixed=False):
	labels, images = readData("train-labels.idx1-ubyte", "train-images.idx3-ubyte", True)
	test_labels, test_images = readData("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", False)
	
	if mode == "all":
		encoding = np.arange(10)
		
		input = images
		teacher = np.repeat(np.array([labels]).T, 10, axis=1)
		teacher = np.equal(teacher, encoding).astype(int)
		
		test_input = test_images
		test_teacher = np.repeat(np.array([test_labels]).T, 10, axis=1)
		test_teacher = np.equal(test_teacher, encoding).astype(int)
		
	else:
		second_label = 3 if mode == "2v3" else 8
		
		input = images[np.where(np.isin(labels, [2, second_label]))]
		teacher = labels[np.where(np.isin(labels, [2, second_label]))]
		teacher = np.reshape(teacher, (teacher.shape[0], 1))
		#teacher = 3 - teacher if second_label == 3 else (8 - teacher) / 6
		teacher = (teacher == 2).astype(int)

		test_input = test_images[np.where(np.isin(test_labels, [2, second_label]))]
		test_teacher = test_labels[np.where(np.isin(test_labels, [2, second_label]))]
		test_teacher = np.reshape(test_teacher, (test_teacher.shape[0], 1))
		#test_teacher = 3 - test_teacher if second_label == 3 else (8 - teacher) / 6
		test_teacher = (test_teacher == 2).astype(int)
		
	n1 = int(np.round(input.shape[0]*0.9))
	if holdoutFixed:
		bits = np.loadtxt('test_holdout_set.txt', dtype=int)
	else:
		bits = np.concatenate((np.ones((n1)), np.zeros((input.shape[0] - n1))))
		np.random.shuffle(bits)
	
	train_input = input[np.where(bits == 1)]
	train_teacher = teacher[np.where(bits == 1)]
	holdout_input = input[np.where(bits == 0)]
	holdout_teacher = teacher[np.where(bits == 0)]
	
#	print(train_input.shape, train_teacher.shape)
#	print(holdout_input.shape, holdout_teacher.shape)
#	print(test_input.shape, test_teacher.shape)
	
	return ([train_input, train_teacher, holdout_input, holdout_teacher, test_input, test_teacher], bits)

"""
Return Values:
0  int   epochs
1  list  trainLossArray
2  list  holdoutLossArray
3  list  testLossArray
4  list  trainPercentCorrectArray
5  list  holdoutPercentCorrectArray
6  list  testPercentCorrectArray
7  float minTrainAccuracy
8  float minHoldoutAccuracy
9  float minTestAccuracy
10 float minTrainLoss
11 float minHoldoutLoss
12 float minTestLoss
13 float minWeightLength
14 array minWeights
"""
def doTrain(nn, inputData, earlyStopThreshold=1e9, maxEpochLimit=1e9):
	train_input, train_teacher, holdout_input, holdout_teacher, test_input, test_teacher = inputData
	trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray = [[],[],[],[],[],[]]
	
	if nn.flow[-1][0] == SigmoidLayer():
		problem_type = "sigmoid"
	elif nn.flow[-1][0] == SoftmaxLayer():
		problem_type = "softmax"
	else:
		print("TrainingError: No training implemented for this type of output layer!")
	
	cnt = 0
	r = 0
	lastLoss = 1e9
	minLoss = 1e9
	
	earlyStopTolerate = earlyStopThreshold * 3 if earlyStopThreshold != 1e9 else 0
	
	while r < earlyStopTolerate or cnt < earlyStopThreshold and r < maxEpochLimit:
		r += 1
		nn.train(train_input, train_teacher)
		loss = nn.loss(holdout_input, holdout_teacher)
		trainLossArray.append(nn.loss(train_input, train_teacher))
		holdoutLossArray.append(loss)
		testLossArray.append(nn.loss(test_input, test_teacher))
		
		trainPercentCorrectArray.append(nn.predictionAccuracy(train_input, train_teacher))
		holdoutPercentCorrectArray.append(nn.predictionAccuracy(holdout_input, holdout_teacher))
		testPercentCorrectArray.append(nn.predictionAccuracy(test_input, test_teacher))
		
		if r % 10 == 0:
			print ("=======epoch ",r , ", current loss is", loss,"=======")
			print ("train   accuracy", trainPercentCorrectArray[-1])
			print ("holdout accuracy", holdoutPercentCorrectArray[-1])
			print ("test    accuracy", testPercentCorrectArray[-1])

		if (loss >= lastLoss - 1e-10):
			cnt += 1
		else:
			cnt = 0
		if (loss < minLoss):
			minLoss = loss
			minWeights = nn.getWeights()
			minTrainAccuracy = trainPercentCorrectArray[-1]
			minHoldoutAccuracy = holdoutPercentCorrectArray[-1]
			minTestAccuracy = testPercentCorrectArray[-1]
			minTrainLoss = trainLossArray[-1]
			minHoldoutLoss = holdoutLossArray[-1]
			minTestLoss = testLossArray[-1]
			minWeightLength = np.linalg.norm(minWeights)
		
		lastLoss = loss
		
	nn.setWeights(minWeights)
	print ("=======", r, "epochs in total, minimum loss is", minLoss ,"=======")
	print ("train   accuracy", minTrainAccuracy)
	print ("holdout accuracy", minHoldoutAccuracy)
	print ("test    accuracy", minTestAccuracy)
	
	return [r, trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray, minTrainAccuracy, minHoldoutAccuracy, minTestAccuracy, minTrainLoss, minHoldoutLoss, minTestLoss, minWeightLength, minWeights]

def plotAccuracyVsInitialRates(accuracyAcrossIR, initialRates):
	plot_title = "Q4. Percent accuracy vs. different initial learning rates"
	file_name = "Q4AccuracyOverIR" + timestamp + ".png"
	accuracyAcrossIR = list(np.asarray(accuracyAcrossIR).T)
	plt.plot(initialRates, accuracyAcrossIR[0], label="train")
	plt.plot(initialRates, accuracyAcrossIR[1], label="holdout")
	plt.plot(initialRates, accuracyAcrossIR[2], label="test")
	plt.title(plot_title)
	plt.xlabel("initial learning rates")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotLossVsInitialRates(lossAcrossIR, initialRates):
	plot_title = "Q4. Cross Entropy Loss vs. different initial learning rates"
	file_name = "Q4LossOverIR" + timestamp + ".png"
	lossAcrossIR = list(np.asarray(lossAcrossIR).T)
	plt.plot(initialRates, lossAcrossIR[0], label="train")
	plt.plot(initialRates, lossAcrossIR[1], label="holdout")
	plt.plot(initialRates, lossAcrossIR[2], label="test")
	plt.title(plot_title)
	plt.xlabel("initial learning rates")
	plt.ylabel("loss")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotAccuracy(trainingStats, mode="2v3"):
	epochs, trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray = trainingStats
	if mode == "2v3":
		plot_title = "Q4. 2 vs 3 Sigmoid  Percent Accuracy"
		file_name = "Q4Accuracy23" + timestamp + ".png"
	elif mode == "2v8":
		plot_title = "Q4. 2 vs 8 Sigmoid  Percent Accuracy"
		file_name = "Q4Accuracy28" + timestamp + ".png"
	else:
		plot_title = "Q6. Softmax Percent Accuracy"
		file_name = "Q6Accuracy" + timestamp + ".png"
	plt.plot(range(epochs), trainPercentCorrectArray, label="train")
	plt.plot(range(epochs), holdoutPercentCorrectArray, label="holdout")
	plt.plot(range(epochs), testPercentCorrectArray, label="test")
	plt.title(plot_title)
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotLoss(trainingStats, mode="2v3"):
	epochs, trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray = trainingStats
	if mode == "2v3":
		plot_title = "Q4. 2 vs 3 Sigmoid  Cross Entropy Loss"
		file_name = "Q4Loss23" + timestamp + ".png"
	elif mode == "2v8":
		plot_title = "Q4. 2 vs 8 Sigmoid  Cross Entropy Loss"
		file_name = "Q4Loss28" + timestamp + ".png"
	else:
		plot_title = "Q6. Softmax Cross Entropy Loss"
		file_name = "Q6Loss" + timestamp + ".png"
	plt.plot(range(epochs), trainLossArray, label="train")
	plt.plot(range(epochs), holdoutLossArray, label="holdout")
	plt.plot(range(epochs), testLossArray, label="test")
	plt.title(plot_title)
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotAccuracyVsLambda(accuracyAcrossLambda, log_lambda, regularization_mode):
	plot_title = "Q5. Percent accuracy vs. different λ with " + regularization_mode + " regularization"
	file_name = "Q5AccuracyOverLambda" + regularization_mode + timestamp + ".png"
	accuracyAcrossLambda = list(np.asarray(accuracyAcrossLambda).T)
	plt.plot(log_lambda, accuracyAcrossLambda[0], label="train")
	plt.title(plot_title)
	plt.xlabel("log(λ)")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotLossVsLambda(lossAcrossLambda, log_lambda, regularization_mode):
	plot_title = "Q5. Test error vs. different λ with " + regularization_mode + " regularization"
	file_name = "Q5LossOverLambda" + regularization_mode + timestamp + ".png"
	lossAcrossLambda = list(np.asarray(lossAcrossLambda).T)
	plt.plot(log_lambda, lossAcrossLambda[2], label="test")
	plt.title(plot_title)
	plt.xlabel("log(λ)")
	plt.ylabel("test error")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()
	
def plotWeightLengthsVsLambda(weightLengths, log_lambda, regularization_mode):
	plot_title = "Q5. Weight lengths vs. different λ with " + regularization_mode + " regularization"
	file_name = "Q5WeightLengthsOverLambda" + regularization_mode + timestamp + ".png"
	plt.plot(log_lambda, weightLengths)
	plt.title(plot_title)
	plt.xlabel("log(λ)")
	plt.ylabel("weight lengths")
	plt.savefig(file_name)
	plt.clf()

def displayWeights(weights, file_name):
	file_name = file_name + timestamp + ".png"
	plt.imshow(weights)
	plt.savefig(file_name)
	plt.clf()

def displayWeightsWithDigits(weights, training_input, file_name):
	file_name = file_name + timestamp + ".png"
	plt.imshow(np.hstack((weights, training_input)))
	plt.savefig(file_name)
	plt.clf()

#											 ::
#											:;J7, :,                        ::;7:
#											,ivYi, ,                       ;LLLFS:
#											:iv7Yi                       :7ri;j5PL
#										 ,:ivYLvr                    ,ivrrirrY2X,
#										 :;r@Wwz.7r:                :ivu@kexianli.
#										:iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
#									 ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
#								;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
#							:,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
#								 ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
#						 ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
#					 :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
#					,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
#				,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
#				:      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
#			 ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
#			,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
#		 ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
#		:,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
#		::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
#		i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
#		:      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
#		i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
#		:    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
#		:     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
#		 :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
#					, :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
#							:jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
#					 ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
#					 ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
#				, , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
#				 :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii::: 

"""
Used to try different initial learning rates
"""
def Q4_trial(classes="2v3", holdoutFixed=False):
	(inputData, bits) = preprocessData(classes, holdoutFixed)
	initialRates = np.linspace(3e-5, 5e-4, num=30)
	accuracyAcrossIR, lossAcrossIR = [],[]
	for initialRate in initialRates:
		nn = NN(lossfunction=BinaryCrossEntropyLossFunction(), trainer=AnnealingTrainer(initialRate), predictor=BinaryPredictor())
		nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.000), SigmoidLayer()])
		trainingStats = doTrain(nn, inputData, earlyStopThreshold=3)
		accuracyAcrossIR.append(trainingStats[7:10])
		lossAcrossIR.append(trainingStats[10:13])
	plotAccuracyVsInitialRates(accuracyAcrossIR, initialRates)
	plotLossVsInitialRates(lossAcrossIR, initialRates)

"""
Used to train logistic regression on 2v3 and 2v8 problems
"""
def Q4(classes="2v3", holdoutFixed=False):
	(inputData, bits) = preprocessData(classes, holdoutFixed)
	nn = NN(lossfunction=BinaryCrossEntropyLossFunction(), trainer=AnnealingTrainer(0.00021), predictor=BinaryPredictor())
	nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.0001), SigmoidLayer()])
	trainingStats = doTrain(nn, inputData, earlyStopThreshold=3)
	plotAccuracy(trainingStats[:7], mode=classes)
	plotLoss(trainingStats[:7], mode=classes)
	return trainingStats[14]

"""
Used to compare final weights of the 2v3 and 2v8 problems
"""
def Q4_weights():
	weights23 = Q4(classes="2v3", holdoutFixed=False)
	weights28 = Q4(classes="2v8", holdoutFixed=False)
	displayWeights(weights23[0][:-1].reshape(28,28), "Q4FinalWeights23")
	displayWeights(weights28[0][:-1].reshape(28,28), "Q4FinalWeights28")
	displayWeights((weights23[0][:-1] - weights28[0][:-1]).reshape(28,28), "Q4FinalWeightsDifference")

"""
Used to train logistic regression with regularization
"""
def Q5(classes="2v3", regularization="L1", holdoutFixed=False):
	(inputData, bits) = preprocessData("2v3", holdoutFixed)
	accuracyAcrossLambda, lossAcrossLambda, weightLengths, finalWeights = [],[],[],[]
	log_lambda = range(-5,3)
	for i in log_lambda:
		lambda_value = 10**i
		nn = NN(lossfunction=BinaryCrossEntropyLossFunction(), trainer=AnnealingTrainer(0.00021), predictor=BinaryPredictor(), regularizer=L1Regularizer(lambda_value) if regularization == "L1" else L2Regularizer(lambda_value))
		nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.0001), SigmoidLayer()])
		trainingStats = doTrain(nn, inputData, earlyStopThreshold=3)
		accuracyAcrossLambda.append(trainingStats[7:10])
		lossAcrossLambda.append(trainingStats[10:13])
		weightLengths.append(trainingStats[13])
		finalWeights.append(trainingStats[14])
	plotAccuracyVsLambda(accuracyAcrossLambda, log_lambda, regularization)
	plotLossVsLambda(lossAcrossLambda, log_lambda, regularization)
	plotWeightLengthsVsLambda(weightLengths, log_lambda, regularization)
	for i, weights in enumerate(finalWeights):
		displayWeights(weights[0][:-1].reshape(28,28), "Q5FinalWeightsLambda" + str(i) + regularization)

"""
Used to train softmax regression
"""
def Q6():
	(inputData, bits) = preprocessData("all", holdoutFixed=False)
	nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=AnnealingTrainer(0.0000114, T=0.2), predictor=MaxPredictor(), regularizer=L2Regularizer(0.001))
	nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=10, weightInitialFactor=0.0001), SoftmaxLayer()])
	trainingStats = doTrain(nn, inputData, earlyStopThreshold=7, maxEpochLimit=2000)
	plotAccuracy(trainingStats[:7], mode="all")
	plotLoss(trainingStats[:7], mode="all")
	finalWeights = trainingStats[14][0]
	for digit in range(finalWeights.shape[1]):
		averageImage = np.mean(inputData[0][np.where(np.isin(np.argmax(inputData[1], axis=1), [digit]))], axis=0)
		displayWeightsWithDigits(finalWeights[:-1,digit].reshape(28,28), averageImage.reshape(28,28), "Q6WeightsDigit" + str(digit))
		
Q4_trial()
Q4_weights()
Q5(regularization="L1")
Q5(regularization="L2")
Q6()
