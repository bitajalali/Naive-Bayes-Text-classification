#!/usr/bin/env python3

''' 
Model for common spatial pattern (CSP) feature calculation and classification for EEG data
'''

import numpy as np
import Naive_Bayes as Naive_Bayes
import DataReader as dataClass
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score



class NaiveBayes_Model:  #main ra be do bakhshe load data va run naive bayes taghsim mikonim

    def __init__(self):       #methode constructore classe naaivebayes model ya haman main dar vaghe
        self.data_path = "HAM-Train-Test/"
        self.ngram = 2
        self.discount = 0.1

    def load_data(self):
        # Read Dataset and Extract Most Common Words and Text Representations based on Term Frequencies
        self.dataset = dataClass.DataReader(self.data_path) #objecti az kelase datareader misazim
        self.dataset.read() #methode read ra az in object faramikhanim

    def run_NaiveBayse(self):
        ################### Create Empty Model ##################
        self.naiveBayesModel = Naive_Bayes.Naive_Bayes(self.dataset)
        #######################Training #########################
        if(self.ngram==1):
            self.naiveBayesModel.train_unigram()
        else:
            self.dataset.calculateAllBigrams()
            self.naiveBayesModel.train_bigram(self.discount)
        ######################Classification ####################
        if (self.ngram == 1):
            trainPredicts = self.naiveBayesModel.predict_unigram(self.dataset.trainTexts)
            testPredicts = self.naiveBayesModel.predict_unigram(self.dataset.testTexts)
        else:
            trainPredicts = self.naiveBayesModel.predict_bigram(self.dataset.trainTexts)
            testPredicts = self.naiveBayesModel.predict_bigram(self.dataset.testTexts)

        train_Accuracy = np.mean(trainPredicts==self.dataset.trainLabelsNum)*100
        test_Accuracy = np.mean(testPredicts == self.dataset.testLabelsNum) * 100
        ####################### Measures #########################
        print("Hint: Stop words has been removed which increase accuracy")
        print("Train Accuracy of Naive Bayes :"+ str(train_Accuracy))
        print("Test Accuracy of Naive Bayes :" + str(test_Accuracy))

        CM = confusion_matrix(self.dataset.testLabelsNum,testPredicts)
        print("=========== Confusion Matrix of Test Data ===========")
        print(CM)
        Precisions = precision_score(self.dataset.testLabelsNum, testPredicts, average=None)
        print("=============== Precisions of Test Data =============")
        print(Precisions)
        Precision_Micro = precision_score(self.dataset.testLabelsNum, testPredicts, average='micro')
        print("Precision Micro Average = "+str(Precision_Micro))
        Precision_Macro = precision_score(self.dataset.testLabelsNum, testPredicts, average='macro')
        print("Precision Macro Average = "+str(Precision_Macro))
        Recalls = recall_score(self.dataset.testLabelsNum, testPredicts, average=None)
        print("================== Recall of Test Data ==============")
        print(Recalls)
        Recall_Micro = recall_score(self.dataset.testLabelsNum, testPredicts, average='micro')
        print("Recall Micro Average = "+str(Recall_Micro))
        Recall_Macro = recall_score(self.dataset.testLabelsNum, testPredicts, average='macro')
        print("Recall Macro Average = "+str(Recall_Macro))
        F_Measure = f1_score(self.dataset.testLabelsNum, testPredicts, average=None)
        print("================= F-Measure of Test Data =============")
        print(F_Measure)
        F_Measure_Micro = f1_score(self.dataset.testLabelsNum, testPredicts, average='micro')
        print("F Micro Average = "+str(F_Measure_Micro))
        F_Measure_Macro = f1_score(self.dataset.testLabelsNum, testPredicts, average='macro')
        print("F Macro Average = "+str(F_Measure_Macro))
        return test_Accuracy




def main():
    model = NaiveBayes_Model()  #objecte kelase naivebayes_model ra tarif mikonim va namash ra model migozarim
    # load data
    model.load_data()  #methode load data ra az kelase in object seda mizanim
    success_rate = model.run_NaiveBayse()

if __name__ == '__main__':
    main()
