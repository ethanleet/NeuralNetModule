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
		return gradient * self.output * (1 - self.output)

class CrossEntropyLossFunction:
	def loss(self, output, teacher):
		#print(teacher.shape, output.shape)
		temp = - np.sum(teacher * np.log(output + 1e-100), axis=1)
		#print(temp.shape)
		return temp
		#return -teacher * np.log(output + 1e-100) - (1 - teacher) * np.log(1 - output + 1e-100)
		
	def gradient(self, output, teacher):
		return (1 - teacher) / (1 - output + 1e-100) - teacher / (output + 1e-100)

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
	def __init__(self, stepSize = 0.05, T = 10):
		self.nstep = 0
		self.stepSize = stepSize
		self.T = T
	
	def train(self, layer, regularizer):
		if hasattr(layer, 'weight'):
			layer.weight -= regularizer.regularize(layer) * (self.stepSize / (1 + self.nstep / self.T))

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
	def __init__(self, lossfunction = CrossEntropyLossFunction(), trainer = NaiveTrainer(), regularizer = NoRegularizer(), predictor = ContinuousPredictor()):
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
		#return np.sum(self.lossfunction.loss(self.test(input), teacher)) / input.shape[0]
		return np.mean(self.lossfunction.loss(self.test(input), teacher))
	
	def predict(self, input):
		return self.predictor.predict(self.test(input))
	
	# Q: this is wrong for softmax. depends on output layer type?
	def predictionAccuracy(self, input, teacher):
		return 1 - np.sum(np.abs(teacher - self.predict(input))) / input.shape[0]
	
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
		teacher = 3 - teacher if second_label == 3 else (8 - teacher) / 6

		test_input = test_images[np.where(np.isin(test_labels, [2, second_label]))]
		test_teacher = test_labels[np.where(np.isin(test_labels, [2, second_label]))]
		test_teacher = np.reshape(test_teacher, (test_teacher.shape[0], 1))
		test_teacher = 3 - test_teacher if second_label == 3 else (8 - teacher) / 6
		
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

	while cnt < earlyStopThreshold and r < maxEpochLimit:
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
			if problem_type == "sigmoid":
				print ("train   wrong/total", np.sum(np.abs(train_teacher - nn.test(train_input)) > 0.5), train_teacher.shape[0])
				print ("holdout wrong/total", np.sum(np.abs(holdout_teacher - nn.test(holdout_input)) > 0.5), holdout_teacher.shape[0])
				print ("test    wrong/total", np.sum(np.abs(test_teacher - nn.test(test_input)) > 0.5), test_teacher.shape[0])
			else:
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
		
		lastLoss = loss
		minWeightLength = np.linalg.norm(minWeights)
		
		
	nn.setWeights(minWeights)
	print ("=======", r, "epochs in total, minimum loss is", minLoss ,"=======")
	if problem_type == "sigmoid":
		print ("train   wrong/total", np.sum(np.abs(train_teacher - nn.test(train_input)) > 0.5), train_teacher.shape[0])
		print ("holdout wrong/total", np.sum(np.abs(holdout_teacher - nn.test(holdout_input)) > 0.5), holdout_teacher.shape[0])
		print ("test    wrong/total", np.sum(np.abs(test_teacher - nn.test(test_input)) > 0.5), test_teacher.shape[0])
	else:
		print ("train   accuracy", trainPercentCorrectArray[-1])
		print ("holdout accuracy", holdoutPercentCorrectArray[-1])
		print ("test    accuracy", testPercentCorrectArray[-1])
	
	return [r, trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray, minTrainAccuracy, minHoldoutAccuracy, minTestAccuracy, minTrainLoss, minHoldoutLoss, minTestLoss, minWeightLength]

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
	plt.legend(bbox_to_anchor=(1.05, 1.), loc=1, borderaxespad=0)
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
	plt.ylabel("accuracy")
	plt.legend(bbox_to_anchor=(1.05, 1.), loc=1, borderaxespad=0)
	plt.savefig(file_name)
	plt.clf()

def plotAccuracy(trainingStats, mode):
	epochs, trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray = trainingStats
	second_label = "3" if mode == "2v3" else "8"
	plot_title = "Q4. 2 vs " + second_label + " Sigmoid Percent Accuracy"
	file_name = "Q4Accuracy2" + second_label + timestamp + ".png"
	plt.plot(range(epochs), trainPercentCorrectArray, label="train")
	plt.plot(range(epochs), holdoutPercentCorrectArray, label="holdout")
	plt.plot(range(epochs), testPercentCorrectArray, label="test")
	plt.title(plot_title)
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.legend(bbox_to_anchor=(1.05, 1.), loc=1, borderaxespad=0)
	plt.savefig(file_name)
	plt.clf()

def plotLoss(trainingStats, mode):
	epochs, trainLossArray, holdoutLossArray, testLossArray, trainPercentCorrectArray, holdoutPercentCorrectArray, testPercentCorrectArray = trainingStats
	second_label = "3" if mode == "2v3" else "8"
	plot_title = "Q4. 2 vs " + second_label + " Sigmoid  Cross Entropy Loss"
	file_name = "Q4Loss2" + second_label + timestamp + ".png"
	plt.plot(range(epochs), trainLossArray, label="train")
	plt.plot(range(epochs), holdoutLossArray, label="holdout")
	plt.plot(range(epochs), testLossArray, label="test")
	plt.title(plot_title)
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend(bbox_to_anchor=(1.05, 1.), loc=1, borderaxespad=0)
	plt.savefig(file_name)
	plt.clf()

def plotAccuracyVsLambda(accuracyAcrossLambda, log_lambda, regularization_mode):
	plot_title = "Q5. Percent accuracy vs. different λ with " + regularization_mode + " regularization"
	file_name = "Q5AccuracyOverLambda" + regularization_mode + timestamp + ".png"
	accuracyAcrossLambda = list(np.asarray(accuracyAcrossLambda).T)
	plt.plot(log_lambda, accuracyAcrossLambda[2], label="test")
	plt.title(plot_title)
	plt.xlabel("log(λ)")
	plt.ylabel("accuracy")
	plt.legend(bbox_to_anchor=(1.05, 1.), loc=1, borderaxespad=0)
	plt.savefig(file_name)
	plt.clf()
	
def plotWeightLengthsVsLambda(weightLengths, log_lambda, regularization_mode):
	plot_title = "Q5. Weight lengths vs. different λ with " + regularization_mode + " regularization"
	file_name = "Q5WeightLengthsOverLambda" + regularization_mode + timestamp + ".png"
	plt.plot(log_lambda, weightLengths, label="test")
	plt.title(plot_title)
	plt.xlabel("log(λ)")
	plt.ylabel("weight lengths")
	plt.legend(bbox_to_anchor=(1.05, 1.), loc=1, borderaxespad=0)
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
def Q4_trial(classes="2v3", holdoutFixed=True):
	(inputData, bits) = preprocessData(classes, holdoutFixed)
	initialRates = np.linspace(1e-5, 2e-5, num=10)
	accuracyAcrossIR, lossAcrossIR = [],[]
	for initialRate in initialRates:
		nn = NN(trainer=AnnealingTrainer(initialRate), predictor=BinaryPredictor())
		nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.0001), SigmoidLayer()])
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
	nn = NN(trainer=AnnealingTrainer(1.6e-5, T=0.3), predictor=BinaryPredictor())
	nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.0001), SigmoidLayer()])
	trainingStats = doTrain(nn, inputData, earlyStopThreshold=3)
	plotAccuracy(trainingStats[:7], classes)
	plotLoss(trainingStats[:7], classes)

"""
Used to find a good holdout set
"""
def Q4crazy(limit, mode="2v3"):
	(inputData, bits) = preprocessData(mode, holdoutFixed=False)
	nn = NN(trainer=AnnealingTrainer(1.12e-5), predictor=BinaryPredictor())
	nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.0001), SigmoidLayer()])
	trainingStats = doTrain(nn, inputData, earlyStopThreshold=3, maxEpochLimit=limit+10)
	plotAccuracy(trainingStats[:7], mode)
	plotLoss(trainingStats[:7], mode)
	if trainingStats[0] < limit:
		np.savetxt('test_holdout_set.txt', bits, fmt='%d')
		return False
	return True

"""
Used to train logistic regression with regularization
"""
def Q5(classes="2v3", regularization="L1", holdoutFixed=False):
	(inputData, bits) = preprocessData("2v3", holdoutFixed)
	accuracyAcrossLambda, weightLengths = [],[]
	log_lambda = range(-3,5)
	for i in log_lambda:
		lambda_value = 10**i
		nn = NN(trainer=AnnealingTrainer(1.8e-5), predictor=BinaryPredictor(), regularizer=L1Regularizer(lambda_value) if regularization == "L1" else L2Regularizer(lambda_value))
		nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=1, weightInitialFactor=0.0001), SigmoidLayer()])
		trainingStats = doTrain(nn, inputData, earlyStopThreshold=3)
		accuracyAcrossLambda.append(trainingStats[7:10])
		weightLengths.append(trainingStats[13])
	plotAccuracyVsLambda(accuracyAcrossLambda, log_lambda, regularization)
	plotWeightLengthsVsLambda(weightLengths, log_lambda, regularization)

"""
Used to train softmax regression
"""
def Q6():
	(inputData, bits) = preprocessData("all")
	nn = NN(trainer=AnnealingTrainer(5e-6), predictor=MaxPredictor(), regularizer=L2Regularizer(0.1))
	nn.addLinearLayers([FullyConnectedLayer(inputSize=784, outputSize=10, weightInitialFactor=0.0001), SoftmaxLayer()])
	trainingStats = doTrain(nn, inputData, earlyStopThreshold=3)
	
#while Q4crazy(300, "2v3"):
#	print("NOO PLZZZZZ")
#print("FUCK YES")




#Q4_trial(holdoutFixed=True)
#Q4(classes="2v3", holdoutFixed=True)
#Q4(classes="2v8", holdoutFixed=True)
#Q5(regularization="L1", holdoutFixed=True)
#Q5(regularization="L2", holdoutFixed=True)
Q6()
