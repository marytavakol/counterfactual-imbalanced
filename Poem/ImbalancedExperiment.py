import DatasetReader
import Skylines
import Logger
import PRM
import numpy
import sys
import PDTTest
from matplotlib import pyplot as plt

if __name__ == '__main__':
    exptNum = 1
    pool = None 
    if len(sys.argv) > 1:
        exptNum = int(sys.argv[1])

    if len(sys.argv) > 2:
         import pathos.multiprocessing as mp
         pool = mp.ProcessingPool(7)

    if exptNum == 1:
        for name in ['scene', 'yeast', 'rcv1_topics', 'tmc2007']:
            dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = True)
            if name == 'rcv1_topics':
                dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])
            else:
                dataset.loadDataset(corpusName = name)

            svm_scores = []
            crf_scores = []
            crf_expected_scores = []
            logger_scores = []
            logger_map_scores = []
            prm_scores = []
            prm_map_scores = []
            erm_scores = []
            erm_map_scores = []
            poem_scores = []
            poem_map_scores = []
            ermstoch_scores = []
            ermstoch_map_scores = []

            svm_time = []
            crf_time = []
            prm_time = []
            erm_time = []
            poem_time = []
            ermstoch_time = []
            for run in range(10):
                print ("************************RUN ", run)
                
                supervised_dataset = DatasetReader.SupervisedDataset(dataset = dataset, verbose = True)
                supervised_dataset.createTrainValidateSplit(validateFrac = 0.25)
        
                svm = Skylines.SVM(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = None)
                svm_time.append(svm.validate())
                svm_scores.append(svm.test())
        
                crf = Skylines.CRF(dataset = supervised_dataset, tol = 1e-6, minC = -2, maxC = 2, verbose = True, parallel = pool)
                crf_time.append(crf.validate())
                crf_scores.append(crf.test())
                crf_expected_scores.append(crf.expectedTestLoss())

                supervised_dataset.freeAuxiliaryMatrices()
                del supervised_dataset

                streamer = Logger.DataStream(dataset = dataset, verbose = True)
                features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

                subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)
                subsampled_dataset.trainFeatures = features
                subsampled_dataset.trainLabels = labels
                logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = True)
                logger_map_scores.append(logger.crf.test())
                logger_scores.append(logger.crf.expectedTestLoss())

                replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = True)

                features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 4)
                replayed_dataset.trainFeatures = features
                replayed_dataset.trainLabels = labels

                sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset)

                bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = True)

                replayed_dataset.freeAuxiliaryMatrices()  
                del replayed_dataset

                bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
                bandit_dataset.createTrainValidateSplit(validateFrac = 0.25)

                coef = None

                logger.freeAuxiliaryMatrices()  
                del logger
               
                prm = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0, 
                                            minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = True, 
                                            parallel = pool, smartStart = coef)
                prm.calibrateHyperParams()
                prm_time.append(prm.validate())
                prm_map_scores.append(prm.test())
                prm_scores.append(prm.expectedTestLoss())
                
                prm.freeAuxiliaryMatrices()  
                del prm

                erm = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1, 
                                            minClip = 0, maxClip = 0, estimator_type = 'Vanilla', verbose = True, 
                                            parallel = None, smartStart = coef)
                erm.calibrateHyperParams()
                erm_time.append(erm.validate())
                erm_map_scores.append(erm.test())
                erm_scores.append(erm.expectedTestLoss())
               
                erm.freeAuxiliaryMatrices()  
                del erm
 
                maj = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0,
                                            minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = True,
                                            parallel = pool, smartStart = coef)
                maj.calibrateHyperParams()
                poem_time.append(maj.validate())
                poem_map_scores.append(maj.test())
                poem_scores.append(maj.expectedTestLoss())

                maj.freeAuxiliaryMatrices()  
                del maj

                majerm = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = 0, maxV = -1,
                                            minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = True,
                                            parallel = None, smartStart = coef)
                majerm.calibrateHyperParams()
                ermstoch_time.append(majerm.validate())
                ermstoch_map_scores.append(majerm.test())
                ermstoch_scores.append(majerm.expectedTestLoss())

                majerm.freeAuxiliaryMatrices()  
                del majerm

                bandit_dataset.freeAuxiliaryMatrices()  
                del bandit_dataset
 
            dataset.freeAuxiliaryMatrices()  
            del dataset
            
            print ("************** RESULTS FOR ", name)
            print ("SVM", svm_scores)
            print ("CRF", crf_scores)
            print ("CRF Expected", crf_expected_scores)
            print ("LOGGER", logger_scores)
            print ("LOGGER MAP", logger_map_scores)
            print ("PRM", prm_scores)
            print ("PRM MAP", prm_map_scores)
            print ("ERM", erm_scores)
            print ("ERM MAP", erm_map_scores)
            print ("POEM", poem_scores)
            print ("POEM MAP", poem_map_scores)
            print ("ERMSTOCH", ermstoch_scores)
            print ("ERMSTOCH MAP", ermstoch_map_scores)

            print ("SVM time", svm_time)
            print ("CRF time", crf_time)
            print ("PRM time", prm_time)
            print ("ERM time", erm_time)
            print ("POEM time", poem_time)
            print ("ERMSTOCH time", ermstoch_time)
            print ("*******************************")
            sys.stdout.flush()

            print ("############### SUMMARY", name)
                
            svm_obs = PDTTest.ExperimentResult(svm_scores, verbose = False)
            print ("SVM", svm_obs.reportMean())

            crf_obs = PDTTest.ExperimentResult(crf_scores, verbose = False)
            print ("CRF", crf_obs.reportMean())

            crfexpected_obs = PDTTest.ExperimentResult(crf_expected_scores, verbose = False)
            print ("CRF Expected", crfexpected_obs.reportMean())

            logger_obs = PDTTest.ExperimentResult(logger_scores, verbose = False)
            print ("LOGGER", logger_obs.reportMean())

            logger_map_obs = PDTTest.ExperimentResult(logger_map_scores, verbose = False)
            print ("LOGGER MAP", logger_map_obs.reportMean())

            prm_obs = PDTTest.ExperimentResult(prm_scores, verbose = False)
            print ("PRM", prm_obs.reportMean())

            prm_map_obs = PDTTest.ExperimentResult(prm_map_scores, verbose = False)
            print ("PRM MAP", prm_map_obs.reportMean())

            erm_obs = PDTTest.ExperimentResult(erm_scores, verbose = False)
            print ("ERM", erm_obs.reportMean())

            erm_map_obs = PDTTest.ExperimentResult(erm_map_scores, verbose = False)
            print ("ERM MAP", erm_map_obs.reportMean())

            poem_obs = PDTTest.ExperimentResult(poem_scores, verbose = False)
            print ("POEM", poem_obs.reportMean())

            poem_map_obs = PDTTest.ExperimentResult(poem_map_scores, verbose = False)
            print ("POEM MAP", poem_map_obs.reportMean())

            ermstoch_obs = PDTTest.ExperimentResult(ermstoch_scores, verbose = False)
            print ("ERMSTOCH", ermstoch_obs.reportMean())

            ermstoch_map_obs = PDTTest.ExperimentResult(ermstoch_map_scores, verbose = False)
            print ("ERMSTOCH MAP", ermstoch_map_obs.reportMean())

            if prm_obs.testDifference(logger_obs):
                print ("PRM > LOGGER")
            else:
                print ("PRM == LOGGER")

            if erm_obs.testDifference(logger_obs):
                print ("ERM > LOGGER")
            else:
                print ("ERM == LOGGER")

            if poem_obs.testDifference(logger_obs):
                print ("POEM > LOGGER")
            else:
                print ("POEM == LOGGER")

            if ermstoch_obs.testDifference(logger_obs):
                print ("ERMSTOCH > LOGGER")
            else:
                print ("ERMSTOCH == LOGGER")

            if poem_obs.testDifference(erm_obs):
                print ("POEM > ERM")
            else:
                print ("POEM == ERM")

            if poem_obs.testDifference(ermstoch_obs):
                print ("POEM > ERMSTOCH")
            else:
                print ("POEM == ERMSTOCH")

            if poem_obs.testDifference(prm_obs):
                print ("POEM > PRM")
            else:
                print ("POEM == PRM")

            if prm_obs.testDifference(erm_obs):
                print ("PRM > ERM")
            else:
                print ("PRM == ERM")
            

            svm_time = PDTTest.ExperimentResult(svm_time, verbose = False)
            print ("SVM TIME", svm_time.reportMean())

            crf_time = PDTTest.ExperimentResult(crf_time, verbose = False)
            print ("CRF TIME", crf_time.reportMean())

            prm_time = PDTTest.ExperimentResult(prm_time, verbose = False)
            print ("PRM TIME", prm_time.reportMean())

            erm_time = PDTTest.ExperimentResult(erm_time, verbose = False)
            print ("ERM TIME", erm_time.reportMean())

            poem_time = PDTTest.ExperimentResult(poem_time, verbose = False)
            print ("POEM TIME", poem_time.reportMean())

            ermstoch_time = PDTTest.ExperimentResult(ermstoch_time, verbose = False)
            print ("ERMSTOCH TIME", ermstoch_time.reportMean())
            
            sys.stdout.flush()
    elif exptNum == 2:
        dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = False)
        dataset.loadDataset(corpusName = 'yeast')

        res = []
        res_costSensitive = []
        imbalance_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        for imbalance_ratio in imbalance_ratios:
            poem_scores = []
            poem_costSensitive_scores = []

            for run in range(5):
                print ("************************RUN ", run)

                streamer = Logger.DataStream(dataset = dataset, verbose = False)
                features, labels = streamer.generateStream(subsampleFrac = 0.05, replayCount = 1)

                subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
                subsampled_dataset.trainFeatures = features
                subsampled_dataset.trainLabels = labels
                logger = Logger.Logger(subsampled_dataset, loggerC = 1, stochasticMultiplier = 1, verbose = False)

                coef = None

                ########
                replayed_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
                features, labels = streamer.generateStream(subsampleFrac = 1.0, replayCount = 1)
                replayed_dataset.trainFeatures = features
                replayed_dataset.trainLabels = labels

                sampledLabels, sampledLogPropensity, sampledLoss, sampledLoss_costSensitive, im_features, im_labels = logger.generateImbalancedLog(replayed_dataset, imbalance_ratio)
                # plt.figure()
                # fig, axes = plt.subplots(nrows=1, ncols=2)
                # ax1, ax2 = axes.flatten()
                # ax1.hist(sampledLoss_costSensitive)
                # ax2.hist(sampledLoss)
                # plt.show()

                replayed_dataset.trainFeatures = im_features
                replayed_dataset.trainLabels = im_labels
                bandit_dataset = DatasetReader.BanditDataset(dataset=replayed_dataset, verbose=False)

                bandit_dataset_costSensitive = DatasetReader.BanditDataset(dataset=replayed_dataset, verbose=False)

                replayed_dataset.freeAuxiliaryMatrices()
                del replayed_dataset

                bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
                bandit_dataset.createTrainValidateSplit(validateFrac = 0.25)

                bandit_dataset_costSensitive.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss_costSensitive)
                bandit_dataset_costSensitive.createTrainValidateSplit(validateFrac=0.25)

                maj = Skylines.PRMWrapper(bandit_dataset, n_iter = 1000, tol = 1e-6, minC = 0, maxC = -1, minV = -6, maxV = 0,
                                            minClip = 0, maxClip = 0, estimator_type = 'Stochastic', verbose = False,
                                            parallel = pool, smartStart = coef)
                maj.calibrateHyperParams()
                maj.validate()


                map_score = maj.test()
                score = maj.expectedTestLoss()
                poem_scores.append(score)

                maj.freeAuxiliaryMatrices()
                del maj

                bandit_dataset.freeAuxiliaryMatrices()
                del bandit_dataset

                maj_costSensitive = Skylines.PRMWrapper(bandit_dataset_costSensitive, n_iter=1000, tol=1e-6, minC=0, maxC=-1, minV=-6, maxV=0,
                                          minClip=0, maxClip=0, estimator_type='Stochastic', verbose=False,
                                          parallel=pool, smartStart=coef)
                maj_costSensitive.calibrateHyperParams()
                maj_costSensitive.validate()

                map_score_costSensitive = maj_costSensitive.test()
                score_costSensitive = maj_costSensitive.expectedTestLoss()
                poem_costSensitive_scores.append(score_costSensitive)

                maj_costSensitive.freeAuxiliaryMatrices()
                del maj_costSensitive

                bandit_dataset_costSensitive.freeAuxiliaryMatrices()
                del bandit_dataset_costSensitive



            poem_obs = PDTTest.ExperimentResult(poem_scores, verbose = False)
            res.append(poem_obs.reportMean())
            poem_mult_obs = PDTTest.ExperimentResult(poem_costSensitive_scores, verbose=False)
            res_costSensitive.append(poem_mult_obs.reportMean())


        print("POEM", res)
        print("POEM costSensitive", res_costSensitive)
        sys.stdout.flush()