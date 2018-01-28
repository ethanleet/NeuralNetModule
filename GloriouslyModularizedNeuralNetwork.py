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
		self.weight = np.random.normal(loc=0, scale=weightInitialFactor / np.sqrt(inputSize + 1), size=(inputSize + 1, outputSize))
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

class TanhLayer:
	def __init__(self, leakFactor = 0.001):
		self.leakFactor = leakFactor
	
	def forward(self, input):
		self.input = input
		return 1.7159 * np.tanh(2/3. * input) + self.leakFactor * input
	def backward(self, gradient):
		return gradient * ((1.7159 * 2)/(3. * np.power(np.cosh(2/3. * self.input), 2)) + self.leakFactor)

class BranchLayer:
	def setTarget(self, targetLayer):
		self.targetLayer = targetLayer
		targetLayer.sourceLayer = self
		
	def forward(self, input):
		self.targetLayer.secondInput = np.copy(input)
		return input
		
	def backward(self, gradient):
		return self.secondGradient + gradient

class ConcatenateLayer:
	def setSource(self, sourceLayer):
		self.sourceLayer = sourceLayer
		sourceLayer.targetLayer = self
		
	def forward(self, input):
		return np.concatenate((input, self.secondInput), axis=1)
		
	def backward(self, gradient):
		self.sourceLayer.secondGradient = gradient[:,-self.secondInput.shape[1]:]
		return gradient[:,:-self.secondInput.shape[1]]

class SoftmaxLayer:
	def __init__(self):
		self.type = "softmax"
	
	def __eq__(self, other): 
		return self.type == other.type
		
	def forward(self, input):
		maxinput = np.reshape(np.amax(input, axis=1), (input.shape[0], 1))
		input -= np.repeat(maxinput, input.shape[1], axis=1)
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
		temp = teacher * np.log(output + 1e-100)
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
	def train(self, layer, regularizer, stepSize):
		if hasattr(layer, 'weight'):
			layer.weight -= regularizer.regularize(layer) * stepSize

class MomentumTrainer:
	def __init__(self, momentumFactor = 0.9):
		self.momentumFactor = momentumFactor
	
	def train(self, layer, regularizer, stepSize):
		if hasattr(layer, 'weight'):
			if hasattr(layer, 'momentum'):
				layer.momentum = layer.momentum * self.momentumFactor + regularizer.regularize(layer) * stepSize
			else:
				layer.momentum = regularizer.regularize(layer) * stepSize
			layer.weight -= layer.momentum

class NesterovTrainer:
	def __init__(self, momentumFactor = 0.9):
		self.momentumFactor = momentumFactor
	
	def train(self, layer, regularizer, stepSize):
		if hasattr(layer, 'weight'):
			if hasattr(layer, 'momentum'):
				momentum_prev = np.copy(layer.momentum)
				layer.momentum = layer.momentum * self.momentumFactor - regularizer.regularize(layer) * stepSize
				layer.weight += -momentum_prev * self.momentumFactor + layer.momentum * (1 + self.momentumFactor)
			else:
				layer.momentum = -regularizer.regularize(layer) * stepSize
				layer.weight += layer.momentum
			
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
		self.numWeights = 0
	
	def addLinearLayers(self, layerList):
		for i in range(len(layerList)):
			if i == len(layerList) - 1:
				self.flow.append((layerList[i], []))
			else:
				self.flow.append((layerList[i], [layerList[i + 1]]))
			if hasattr(layerList[i], 'weight'):
				self.numWeights += layerList[i].weight.size
		
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
		return np.sum(np.sum(np.abs(teacher != self.predict(input)), axis=1) == 0) / float(input.shape[0])
	
	def train(self, input, teacher, stepSize):
		output = self.test(input)
		gradient = self.lossfunction.gradient(output, teacher)
		for step in reversed(self.flow):
			layer = step[0]
			#Assume to be linear
			gradient = layer.backward(gradient)
			self.trainer.train(layer, self.regularizer, stepSize / np.sqrt(self.numWeights))
	
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
				layer.weight = np.copy(weights[i])
				i += 1

class Data:
	def __init__(self, training_input, training_teacher, holdout_input, holdout_teacher, test_input, test_teacher):
		self.train_input = training_input
		self.train_teacher = training_teacher
		self.holdout_input = holdout_input
		self.holdout_teacher = holdout_teacher
		self.test_input = test_input
		self.test_teacher = test_teacher

class BatchTrainingMethod:
	def train(self, nn, input, teacher, stepSize):
		nn.train(input, teacher, stepSize)

class MiniBatchTrainingMethod:
	def __init__(self, batchSize=128):
		self.batchSize = batchSize
	
	def train(self, nn, input, teacher, stepSize):
		permutation = np.random.permutation(input.shape[0])
		input = input[permutation]
		teacher = teacher[permutation]
		
		for i in range(np.ceil(input.shape[0] / float(self.batchSize)).astype(int)):
			batch_input = input[i * self.batchSize: (i+1) * self.batchSize]
			batch_teacher = teacher[i * self.batchSize: (i+1) * self.batchSize]
			nn.train(batch_input, batch_teacher, stepSize)

class PowerAnnealingFunction:
	def __init__(self, initialStepSize = 0.05, T = 0.1, power = 0.5):
		self.nstep = 0
		self.initialStepSize = initialStepSize
		self.T = T
		self.power = power
	
	def evaluate(self, stepNum):
		return self.initialStepSize / (1 + np.power(self.nstep, self.power) / self.T)

class NNTrainingWorkflow:
	def __init__(self, nn, data, timeout = 1000, trainingMethod = BatchTrainingMethod(), annealingFunction = PowerAnnealingFunction(), callbackFunction = None):
		self.nn = nn
		self.data = data
		self.timeout = timeout
		self.trainingMethod = trainingMethod
		self.annealingFunction = annealingFunction
		self.callbackFunction = callbackFunction
		self.earlyStop = False
	
	def train(self):
		self.t = 0
		while self.t < self.timeout and not self.earlyStop:
			stepSize = self.annealingFunction.evaluate(self.t)
			self.trainingMethod.train(self.nn, self.data.train_input, self.data.train_teacher, stepSize)
			if self.callbackFunction is not None:
				self.callbackFunction(self)
			self.t += 1

def readData(label_fl, image_file, training):
	
	'''
	self function is used to read in data from MNIST data set
	'''
	
	val = 60000 if training else 6000
	label_file =  open(label_fl, 'rb')
	tup = struct.unpack(">II", label_file.read(8))
	labels = np.fromfile(label_file, dtype=np.int8)
	
	img_file = open(image_file, 'rb')
	tup = struct.unpack(">IIII", img_file.read(16))
	images = np.fromfile(img_file, dtype=np.uint8).reshape(len(labels), tup[2], tup[3])
	
	subset = np.zeros((val,tup[2]*tup[3]),dtype=np.float64)
	if training:
		for img in range(val):
			subset[img] = images[img].flatten()/float(127.5) - 1.0
		return (labels[:val],subset)
	else:
		for img in range(val):
			subset[img] = images[len(images) - val + img].flatten()/float(127.5) - 1.0
		return (labels[len(images) - val:],subset)

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
		teacher = (teacher == 2).astype(int)

		test_input = test_images[np.where(np.isin(test_labels, [2, second_label]))]
		test_teacher = test_labels[np.where(np.isin(test_labels, [2, second_label]))]
		test_teacher = np.reshape(test_teacher, (test_teacher.shape[0], 1))
		test_teacher = (test_teacher == 2).astype(int)
	
	# Holdout size
	n1 = 10000
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

def callback3e(wf):
	if not hasattr(wf, "trainLossArray"):
		wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, wf.trainPercentCorrectArray, wf.holdoutPercentCorrectArray, wf.testPercentCorrectArray = [[],[],[],[],[],[]]
		wf.cnt = 0
		wf.lastLoss = 1e9
		wf.minLoss = 1e9
		
	nn = wf.nn
	loss = nn.loss(wf.data.holdout_input, wf.data.holdout_teacher)
	wf.trainLossArray.append(nn.loss(wf.data.train_input, wf.data.train_teacher))
	wf.holdoutLossArray.append(loss)
	wf.testLossArray.append(nn.loss(wf.data.test_input, wf.data.test_teacher))
		
	wf.trainPercentCorrectArray.append(nn.predictionAccuracy(wf.data.train_input, wf.data.train_teacher))
	wf.holdoutPercentCorrectArray.append(nn.predictionAccuracy(wf.data.holdout_input, wf.data.holdout_teacher))
	wf.testPercentCorrectArray.append(nn.predictionAccuracy(wf.data.test_input, wf.data.test_teacher))
		
	if wf.t % 1 == 0:
		print ("=======epoch ",wf.t , ", current loss is", loss,"=======")
		print ("train   accuracy", wf.trainPercentCorrectArray[-1])
		print ("holdout accuracy", wf.holdoutPercentCorrectArray[-1])
		print ("test    accuracy", wf.testPercentCorrectArray[-1])

	if (loss >= wf.lastLoss - 1e-10):
		wf.cnt += 1
	else:
		wf.cnt = 0
	if (loss < wf.minLoss):
		wf.minLoss = loss
		wf.minWeights = nn.getWeights()
		wf.minT = wf.t
	wf.lastLoss = loss

	if wf.t > 10 and wf.cnt >= 3:
		wf.earlyStop = True
	
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
	plot_title = "Q5. Percent accuracy vs. different lambda with " + regularization_mode + " regularization"
	file_name = "Q5AccuracyOverLambda" + regularization_mode + timestamp + ".png"
	accuracyAcrossLambda = list(np.asarray(accuracyAcrossLambda).T)
	plt.plot(log_lambda, accuracyAcrossLambda[0], label="train")
	plt.title(plot_title)
	plt.xlabel("log(lambda)")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotLossVsLambda(lossAcrossLambda, log_lambda, regularization_mode):
	plot_title = "Q5. Test error vs. different lambda with " + regularization_mode + " regularization"
	file_name = "Q5LossOverLambda" + regularization_mode + timestamp + ".png"
	lossAcrossLambda = list(np.asarray(lossAcrossLambda).T)
	plt.plot(log_lambda, lossAcrossLambda[2], label="test")
	plt.title(plot_title)
	plt.xlabel("log(lambda)")
	plt.ylabel("test error")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()
	
def plotWeightLengthsVsLambda(weightLengths, log_lambda, regularization_mode):
	plot_title = "Q5. Weight lengths vs. different lambda with " + regularization_mode + " regularization"
	file_name = "Q5WeightLengthsOverLambda" + regularization_mode + timestamp + ".png"
	plt.plot(log_lambda, weightLengths)
	plt.title(plot_title)
	plt.xlabel("log(lambda)")
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
	trainingStats = doTrain(nn, inputData, earlyStopThreshold=3, maxEpochLimit=2000)
	plotAccuracy(trainingStats[:7], mode="all")
	plotLoss(trainingStats[:7], mode="all")
	finalWeights = trainingStats[14][0]
	for digit in range(finalWeights.shape[1]):
		averageImage = np.mean(inputData[0][np.where(np.isin(np.argmax(inputData[1], axis=1), [digit]))], axis=0)
		displayWeightsWithDigits(finalWeights[:-1,digit].reshape(28,28), averageImage.reshape(28,28), "Q6WeightsDigit" + str(digit))


# Metaparameter
nHiddenUnits = 64
epsilon = 0.001

# Read data
(inputData, bits) = preprocessData("all", holdoutFixed=False)
data = Data(inputData[0], inputData[1], inputData[2], inputData[3], inputData[4], inputData[5])


# Check with numerical approximation
nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NaiveTrainer(), predictor=MaxPredictor(), regularizer=NoRegularizer())
nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), SigmoidLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])


def numericApprox(nn, layerNum, rowNum, colNum, graphID):
	oldWeight = nn.getWeights()
	
	weightPlus = np.copy(nn.getWeights())
	weightMinus = np.copy(nn.getWeights())
	weightPlus[layerNum][rowNum,colNum] += epsilon
	nn.setWeights(weightPlus)
	lossPlus = nn.loss(inputData[0][graphID:graphID+1], inputData[1][graphID:graphID+1])
	
	weightMinus[layerNum][rowNum,colNum] -= epsilon
	nn.setWeights(weightMinus)
	lossMinus = nn.loss(inputData[0][graphID:graphID+1], inputData[1][graphID:graphID+1])
	
	nn.setWeights(np.copy(oldWeight))
	nn.train(inputData[0][graphID:graphID+1], inputData[1][graphID:graphID+1], 1)
	newWeight = nn.getWeights()
	
	gradient = np.sqrt(nn.numWeights)*(oldWeight[layerNum][rowNum,colNum] - newWeight[layerNum][rowNum,colNum])
	approx = (lossPlus - lossMinus) / (2 * epsilon)
	print(gradient, approx, abs(gradient-approx)/(epsilon**2))
	nn.setWeights(oldWeight)

#numericApprox(nn, 0, 0, 0, 0)
#numericApprox(nn, 1, 0, 0, 0)
#numericApprox(nn, 0, 784, 0, 0)
#numericApprox(nn, 1, 64, 0, 0)
# Train Q3
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NaiveTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.001))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), SigmoidLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#workflow = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=1, T=0.2), callbackFunction=callback3e)
#workflow.train()


#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NesterovTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.01))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])

#workflow = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.05, T=0.5), callbackFunction=callback3e)
#workflow.train()

branchLayer1 = BranchLayer()
fcLayer1 = FullyConnectedLayer(inputSize=784, outputSize=64, weightInitialFactor=1)
tanhLayer1 = TanhLayer()
concatenateLayer1 = ConcatenateLayer()
branchLayer1.setTarget(concatenateLayer1)
branchLayer2 = BranchLayer()
fcLayer2 = FullyConnectedLayer(inputSize=784+64, outputSize=32, weightInitialFactor=1)
tanhLayer2 = TanhLayer()
concatenateLayer2 = ConcatenateLayer()
branchLayer2.setTarget(concatenateLayer2)
branchLayer3 = BranchLayer()
fcLayer3 = FullyConnectedLayer(inputSize=784+64+32, outputSize=32, weightInitialFactor=1)
tanhLayer3 = TanhLayer()
concatenateLayer3 = ConcatenateLayer()
branchLayer3.setTarget(concatenateLayer3)
fcLayer4 = FullyConnectedLayer(inputSize=784+64+32+32, outputSize=10, weightInitialFactor=1)
softmaxLayer = SoftmaxLayer()


nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NesterovTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.01))
nn.addLinearLayers([branchLayer1, fcLayer1, tanhLayer1, concatenateLayer1, branchLayer2, fcLayer2, tanhLayer2, concatenateLayer2, branchLayer3, fcLayer3, tanhLayer3, concatenateLayer3, fcLayer4, softmaxLayer])

workflow = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.05, T=0.5), callbackFunction=callback3e)
workflow.train()
