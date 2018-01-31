# Author: Chao Zhang, Yiting Li

from __future__ import print_function
import datetime
import matplotlib.pyplot as plt
import numpy as np
import struct
import sys

_python_version = sys.version_info[0]
timestamp = ''.join(c for c in datetime.datetime.now().isoformat() if c.isdigit())

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

#class ConcatenateLayer:
#	def setSource(self, sourceLayer):
#		self.sourceLayer = sourceLayer
#		sourceLayer.targetLayer = self
#		
#	def forward(self, input):
#		return np.concatenate((input, self.secondInput), axis=1)
#		
#	def backward(self, gradient):
#		self.sourceLayer.secondGradient = gradient[:,-self.secondInput.shape[1]:]
#		return gradient[:,:-self.secondInput.shape[1]]

class ConcatenateLayer:
	def setSource(self, sourceLayer):
		self.sourceLayer = sourceLayer
		sourceLayer.targetLayer = self
		
	def forward(self, input):
		return np.concatenate((input, self.secondInput), axis=1)
		
	def backward(self, gradient):
		temp, self.sourceLayer.secondGradient = np.split(gradient, [gradient.shape[1] - self.secondInput.shape[1]], axis=1)
		return temp

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
		self.t = 0 # epoch count
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
	print("gradient =", gradient)
	print("approx.  =", approx)
	print("ratio    =", abs(gradient-approx)/(epsilon**2), "\n")
	nn.setWeights(oldWeight)

def callback3e(wf):
	if not hasattr(wf, "trainLossArray"):
		wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray = [[],[],[],[],[],[]]
		wf.cnt = 0
		wf.lastLoss = 1e9
		wf.minLoss = 1e9
		
	nn = wf.nn
	loss = nn.loss(wf.data.holdout_input, wf.data.holdout_teacher)
	wf.trainLossArray.append(nn.loss(wf.data.train_input, wf.data.train_teacher))
	wf.holdoutLossArray.append(loss)
	wf.testLossArray.append(nn.loss(wf.data.test_input, wf.data.test_teacher))
		
	wf.trainAccuracyArray.append(nn.predictionAccuracy(wf.data.train_input, wf.data.train_teacher))
	wf.holdoutAccuracyArray.append(nn.predictionAccuracy(wf.data.holdout_input, wf.data.holdout_teacher))
	wf.testAccuracyArray.append(nn.predictionAccuracy(wf.data.test_input, wf.data.test_teacher))
		
	if wf.t % 1 == 0:
		print ("=======epoch ",wf.t , ", current loss is", loss,"=======")
		print ("train   accuracy", wf.trainAccuracyArray[-1])
		print ("holdout accuracy", wf.holdoutAccuracyArray[-1])
		print ("test    accuracy", wf.testAccuracyArray[-1])

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

def plotAccuracy(nEpochs, trainAccuracyArray, holdoutAccuracyArray, testAccuracyArray, mode="Q3"):
	plot_title = mode + " Softmax Percent Accuracy"
	file_name = timestamp + "_" + mode + "_Accuracy.png"
	plt.plot(range(nEpochs), trainAccuracyArray, label="train")
	plt.plot(range(nEpochs), holdoutAccuracyArray, label="holdout")
	plt.plot(range(nEpochs), testAccuracyArray, label="test")
	plt.title(plot_title)
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.legend()
	plt.savefig(file_name)
	plt.clf()

def plotLoss(nEpochs, trainLossArray, holdoutLossArray, testLossArray, mode="Q3"):
	plot_title = mode + " Softmax Cross Entropy Loss"
	file_name = timestamp + "_" + mode + "_Loss.png"
	plt.plot(range(nEpochs), trainLossArray, label="train")
	plt.plot(range(nEpochs), holdoutLossArray, label="holdout")
	plt.plot(range(nEpochs), testLossArray, label="test")
	plt.title(plot_title)
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend()
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

#                       ::
#                      :;J7, :,                        ::;7:
#                      ,ivYi, ,                       ;LLLFS:
#                      :iv7Yi                       :7ri;j5PL
#                     ,:ivYLvr                    ,ivrrirrY2X,
#                     :;r@Wwz.7r:                :ivu@kexianli.
#                    :iL7::,:::iiirii:ii;::::,,irvF7rvvLujL7ur
#                   ri::,:,::i:iiiiiii:i:irrv177JX7rYXqZEkvv17
#                ;i:, , ::::iirrririi:i:::iiir2XXvii;L8OGJr71i
#              :,, ,,:   ,::ir@mingyi.irii:i:::j1jri7ZBOS7ivv,
#                 ,::,    ::rv77iiiriii:iii:i::,rvLq@huhao.Li
#             ,,      ,, ,:ir7ir::,:::i;ir:::i:i::rSGGYri712:
#           :::  ,v7r:: ::rrv77:, ,, ,:i7rrii:::::, ir7ri7Lri
#          ,     2OBBOi,iiir;r::        ,irriiii::,, ,iv7Luur:
#        ,,     i78MBBi,:,:::,:,  :7FSL: ,iriii:::i::,,:rLqXv::
#        :      iuMMP: :,:::,:ii;2GY7OBB0viiii:i:iii:i:::iJqL;::
#       ,     ::::i   ,,,,, ::LuBBu BBBBBErii:i:i:i:i:i:i:r77ii
#      ,       :       , ,,:::rruBZ1MBBqi, :,,,:::,::::::iiriri:
#     ,               ,,,,::::i:  @arqiao.       ,:,, ,:::ii;i7:
#    :,       rjujLYLi   ,,:::::,:::::::::,,   ,:i,:,,,,,::i:iii
#    ::      BBBBBBBBB0,    ,,::: , ,:::::: ,      ,,,, ,,:::::::
#    i,  ,  ,8BMMBBBBBBi     ,,:,,     ,,, , ,   , , , :,::ii::i::
#    :      iZMOMOMBBM2::::::::::,,,,     ,,,,,,:,,,::::i:irr:i:::,
#    i   ,,:;u0MBMOG1L:::i::::::  ,,,::,   ,,, ::::::i:i:iirii:i:i:
#    :    ,iuUuuXUkFu7i:iii:i:::, :,:,: ::::::::i:i:::::iirr7iiri::
#    :     :rk@Yizero.i:::::, ,:ii:::::::i:::::i::,::::iirrriiiri::,
#     :      5BMBBBBBBSr:,::rv2kuii:::iii::,:i:,, , ,,:,:i@petermu.,
#          , :r50EZ8MBBBBGOBBBZP7::::i::,:::::,: :,:,::i;rrririiii::
#              :jujYY7LS0ujJL7r::,::i::,::::::::::::::iirirrrrrrr:ii:
#           ,:  :@kevensun.:,:,,,::::i:i:::::,,::::::iir;ii;7v77;ii;i,
#           ,,,     ,,:,::::::i:iiiii:i::::,, ::::iiiir@xingjief.r;7:i,
#        , , ,,,:,,::::::::iiiiiiiiii:,:,:::::::::iiir;ri7vL77rrirri::
#         :,, , ::::::::i:::i:::i:i::,,,,,:,::i:i:::iir;@Secbone.ii:::

"""Metaparameter"""
nHiddenUnits = 64
epsilon = 0.001

"""Read data"""
(inputData, bits) = preprocessData("all", holdoutFixed=False)
data = Data(inputData[0], inputData[1], inputData[2], inputData[3], inputData[4], inputData[5])

"""Check with numerical approximation"""
nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NaiveTrainer(), predictor=MaxPredictor(), regularizer=NoRegularizer())
nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), SigmoidLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])

numericApprox(nn, 0, 0, 0, 0)
numericApprox(nn, 0, 32, 32, 0)
numericApprox(nn, 1, 0, 0, 0)
numericApprox(nn, 1, 32, 2, 0)
numericApprox(nn, 0, 784, 0, 0)
numericApprox(nn, 1, 64, 0, 0)
numericApprox(nn, 0, 0, 0, 3)
numericApprox(nn, 0, 32, 32, 3)
numericApprox(nn, 1, 0, 0, 3)
numericApprox(nn, 1, 32, 2, 3)
numericApprox(nn, 0, 784, 0, 3)
numericApprox(nn, 1, 64, 0, 3)

"""Train Q3"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NaiveTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=np.sqrt(785)*0.01), SigmoidLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=np.sqrt(64)*0.01), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.5, T=0.2), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "Q3")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "Q3")

"""Train tanh"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NaiveTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=np.sqrt(785)*0.01), TanhLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=np.sqrt(64)*0.01), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.1, T=0.2), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "tanh")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "tanh")

"""Train weight initialization"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NaiveTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.1, T=0.1), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "Init")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "Init")

"""Train momentum"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=MomentumTrainer(momentumFactor=0.9), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.01, T=0.5), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "Momentum")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "Momentum")

"""Train 128 hidden units"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=MomentumTrainer(momentumFactor=0.9), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=128, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=128, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.01, T=0.5), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "128HU")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "128HU")

"""Train 32 hidden units"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=MomentumTrainer(momentumFactor=0.9), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=32, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=32, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.01, T=0.5), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "32HU")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "32HU")

"""Train 2 hidden layers"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=MomentumTrainer(momentumFactor=0.9), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=60, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=60, outputSize=60, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=60, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.005, T=0.8), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "2HL")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "2HL")

"""Train Nesterov momentum"""
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NesterovTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=nHiddenUnits, weightInitialFactor=1), TanhLayer(), FullyConnectedLayer(inputSize=nHiddenUnits, outputSize=10, weightInitialFactor=1), SoftmaxLayer()])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.02, T=0.5), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "Nesterov")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "Nesterov")

"""Train weird architecture"""
#branchLayer1 = BranchLayer()
#fcLayer1 = FullyConnectedLayer(inputSize=784, outputSize=64, weightInitialFactor=1)
#tanhLayer1 = TanhLayer()
#concatenateLayer1 = ConcatenateLayer()
#branchLayer1.setTarget(concatenateLayer1)
#fcLayer2 = FullyConnectedLayer(inputSize=784+64, outputSize=10, weightInitialFactor=1)
#softmaxLayer = SoftmaxLayer()
#
#nn = NN(lossfunction=MultiwayCrossEntropyLossFunction(), trainer=NesterovTrainer(), predictor=MaxPredictor(), regularizer=L1Regularizer(modifier=0.002))
#nn.addLinearLayers([branchLayer1, fcLayer1, tanhLayer1, concatenateLayer1, fcLayer2, softmaxLayer])
#
#wf = NNTrainingWorkflow(nn, data=data, timeout=1e3, trainingMethod=MiniBatchTrainingMethod(), annealingFunction=PowerAnnealingFunction(initialStepSize=0.02, T=0.5), callbackFunction=callback3e)
#wf.train()
#plotLoss(wf.t, wf.trainLossArray, wf.holdoutLossArray, wf.testLossArray, "Weird")
#plotAccuracy(wf.t, wf.trainAccuracyArray, wf.holdoutAccuracyArray, wf.testAccuracyArray, "Weird")
