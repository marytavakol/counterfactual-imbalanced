import DatasetReader
import math
import numpy
import numpy.random
import scipy.sparse
import Skylines
import sys
from matplotlib import pyplot as plt

class DataStream:
    def __init__(self, dataset, verbose):
        self.verbose = verbose

        self.originalFeatures = dataset.trainFeatures.copy()
        self.originalLabels = dataset.trainLabels.copy()
        
        numSamples = numpy.shape(self.originalFeatures)[0]
        permute = numpy.random.permutation(numSamples)
        self.permutedFeatures = self.originalFeatures[permute, :]
        self.permutedLabels = self.originalLabels[permute, :]

        if self.verbose:
            print ("DataStream: [Message] Initialized with permutation over [n_samples]: ", numSamples)
            sys.stdout.flush()

    def generateStream(self, subsampleFrac, replayCount):
        numSamples = numpy.shape(self.permutedFeatures)[0]
        numSubsamples = math.ceil(subsampleFrac*numSamples)
        subsampleFeatures = self.permutedFeatures[0:numSubsamples,:]
        subsampleLabels = self.permutedLabels[0:numSubsamples,:]
        if self.verbose:
            print ("DataStream: [Message] Selected subsamples [n_subsamples, n_samples]: ", numSubsamples, numSamples)
            sys.stdout.flush()

        if replayCount <= 1: 
            return subsampleFeatures, subsampleLabels
        else:
            replicator = numpy.ones((replayCount, 1))
            repeatedFeatures = scipy.sparse.kron(replicator, subsampleFeatures, format='csr')
            repeatedLabels = numpy.kron(replicator, subsampleLabels)

            if self.verbose:
                print ("DataStream: [Message] Replay samples ", numpy.shape(repeatedFeatures)[0])
                sys.stdout.flush()
            return repeatedFeatures, repeatedLabels

    def freeAuxiliaryMatrices(self):
        del self.originalFeatures
        del self.originalLabels
        del self.permutedFeatures
        del self.permutedLabels
        
        if self.verbose:
            print ("Datastream: [Message] Freed matrices")
            sys.stdout.flush()


class Logger:
    def __init__(self, dataset, loggerC, stochasticMultiplier, verbose):
        self.verbose = verbose
        crf = Skylines.CRF(dataset = dataset, tol = 1e-5, minC = loggerC, maxC = loggerC, verbose = self.verbose, parallel = False)
        crf.Name = "LoggerCRF"
        crf.validate()
        if not(stochasticMultiplier == 1):
            for i in range(len(crf.labeler)):
                if crf.labeler[i] is not None:
                    crf.labeler[i].coef_ = stochasticMultiplier * crf.labeler[i].coef_

        self.crf = crf
        if self.verbose:
            print ("Logger: [Message] Trained logger crf. Weight-scale: ", stochasticMultiplier)
            sys.stdout.flush()

    def freeAuxiliaryMatrices(self):
        del self.crf

    def generateLog(self, dataset):
        numSamples, numFeatures = numpy.shape(dataset.trainFeatures)
        numLabels = numpy.shape(dataset.trainLabels)[1]

        sampledLabels = numpy.zeros((numSamples, numLabels), dtype = numpy.int)
        logpropensity = numpy.zeros(numSamples, dtype = numpy.longdouble)
        for i in range(numLabels):
            if self.crf.labeler[i] is not None:
                regressor = self.crf.labeler[i]
                predictedProbabilities = regressor.predict_log_proba(dataset.trainFeatures)

                randomThresholds = numpy.log(numpy.random.rand(numSamples).astype(numpy.longdouble))
                sampledLabel = randomThresholds > predictedProbabilities[:,0]
                sampledLabels[:, i] = sampledLabel.astype(int)

                probSampledLabel = numpy.zeros(numSamples, dtype=numpy.longdouble)
                probSampledLabel[sampledLabel] = predictedProbabilities[sampledLabel, 1]
                remainingLabel = numpy.logical_not(sampledLabel)
                probSampledLabel[remainingLabel] = predictedProbabilities[remainingLabel, 0]
                logpropensity = logpropensity + probSampledLabel

        diffLabels = sampledLabels != dataset.trainLabels
        sampledLoss = diffLabels.sum(axis = 1, dtype = numpy.longdouble) - numLabels

        if self.verbose:
            averageSampledLoss = sampledLoss.mean(dtype = numpy.longdouble)
            print ("Logger: [Message] Sampled historical logs. [Mean train loss, numSamples]:", averageSampledLoss, numpy.shape(sampledLabels)[0])
            print ("Logger: [Message] [min, max, mean] inv propensity", logpropensity.min(), logpropensity.max(), logpropensity.mean())
            sys.stdout.flush()
        return sampledLabels, logpropensity, sampledLoss



    def generateImbalancedLog(self, dataset, imbalance_ratio):
        numSamples, numFeatures = numpy.shape(dataset.trainFeatures)
        numLabels = numpy.shape(dataset.trainLabels)[1]

        sampledLabels = numpy.zeros((numSamples, numLabels), dtype = numpy.int)
        logpropensity = numpy.zeros(numSamples, dtype = numpy.longdouble)
        for i in range(numLabels):
            if self.crf.labeler[i] is not None:
                regressor = self.crf.labeler[i]
                predictedProbabilities = regressor.predict_log_proba(dataset.trainFeatures)

                randomThresholds = numpy.log(numpy.random.rand(numSamples).astype(numpy.longdouble))
                sampledLabel = randomThresholds > predictedProbabilities[:,0]
                sampledLabels[:, i] = sampledLabel.astype(int)

                probSampledLabel = numpy.zeros(numSamples, dtype=numpy.longdouble)
                probSampledLabel[sampledLabel] = predictedProbabilities[sampledLabel, 1]
                remainingLabel = numpy.logical_not(sampledLabel)
                probSampledLabel[remainingLabel] = predictedProbabilities[remainingLabel, 0]
                logpropensity = logpropensity + probSampledLabel

        diffLabels = sampledLabels != dataset.trainLabels
        sampledLoss = diffLabels.sum(axis = 1, dtype = numpy.longdouble) - numLabels
        # plt.figure()
        # plt.hist(sampledLoss)
        # plt.show()

        threshold = min(sampledLoss)/2
        LargeLossSet = numpy.where(sampledLoss >= threshold)[0]
        SmallLossSet = numpy.where(sampledLoss < threshold)[0]
        num_minor = math.ceil(imbalance_ratio*numSamples)
        if num_minor <= len(LargeLossSet):
            minorSet_index = numpy.random.choice(LargeLossSet, num_minor)
        else:
            repeat = round(num_minor/len(LargeLossSet))
            minorSet_index = numpy.repeat(LargeLossSet, repeat)

        num_minor = len(minorSet_index)

        num_major = numSamples - num_minor
        if num_major <= len(SmallLossSet):
            majorSet_index = numpy.random.choice(SmallLossSet, num_major)
        else:
            repeat = round(num_major/len(SmallLossSet))
            majorSet_index = numpy.repeat(SmallLossSet, repeat)

        newSet_index = numpy.concatenate((minorSet_index,majorSet_index))
        permute = numpy.random.permutation(len(newSet_index))
        imbalanced_index = newSet_index[permute]

        imbalanced_sampledLoss = sampledLoss[imbalanced_index]
        imbalanced_logpropensity = logpropensity[imbalanced_index]
        imbalanced_sampledLabels = sampledLabels[imbalanced_index, :]
        imbalanced_features = dataset.trainFeatures[imbalanced_index, :]
        imbalanced_labels = dataset.trainLabels[imbalanced_index, :]

        CostSensitiveLoss = numpy.copy(imbalanced_sampledLoss)

        for i, val in enumerate(imbalanced_sampledLoss):
            if val < threshold:
                CostSensitiveLoss[i] = val #* (1-imbalance_ratio)
            else:
                CostSensitiveLoss[i] = val * (imbalance_ratio * 2)


        if self.verbose:
            averageSampledLoss = sampledLoss.mean(dtype = numpy.longdouble)
            print ("Logger: [Message] Sampled historical logs. [Mean train loss, numSamples]:", averageSampledLoss, numpy.shape(sampledLabels)[0])
            print ("Logger: [Message] [min, max, mean] inv propensity", logpropensity.min(), logpropensity.max(), logpropensity.mean())
            sys.stdout.flush()
        return imbalanced_sampledLabels, imbalanced_logpropensity, imbalanced_sampledLoss, CostSensitiveLoss, imbalanced_features, imbalanced_labels

        # sorted_loss_index = numpy.argsort(sampledLoss)
        # minorSet_index = sorted_loss_index[-num_minor:]
        #
        # imbalance_distance = math.floor((0.5 - imbalance_ratio) * 10)
        # threshold = min(sampledLoss[minorSet_index]) - imbalance_distance
        # majorSet_index = numpy.where(sampledLoss < threshold)[0]
        # num_major = numSamples - num_minor
        # repeat = round(num_major/len(majorSet_index))
        # majorSet_index = numpy.repeat(majorSet_index, repeat)
        # newSet_index = numpy.concatenate((minorSet_index,majorSet_index))
        # permute = numpy.random.permutation(len(newSet_index))
        # imbalanced_index = newSet_index[permute]