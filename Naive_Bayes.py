#!/usr/bin/env python3

'''	Loads the dataset 2a of the BCI Competition IV
available on http://bnci-horizon-2020.eu/database/data-sets
'''

import numpy as np
import DataReader as Dataset
from collections import Counter

class Naive_Bayes:
    '''
            '''

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.p_classes = np.zeros(self.dataset.labels.__len__())
        self.p_words_classes = np.zeros((self.dataset.numBestWords,self.dataset.labels.__len__()))
        self.discount = 0.1
        self.bigramConditional = np.zeros((self.dataset.labels.__len__(), self.dataset.allBigrams.__len__()))

    def train_unigram(self):
        self.p_words_classes = np.zeros((self.dataset.numBestWords, self.dataset.labels.__len__()))
        # Extract Word Frequencies Representation for Each Text
        trainWordsFrequency = np.zeros((self.dataset.trainTexts.__len__(),self.dataset.numBestWords))
        for i in range(0,self.dataset.trainTexts.__len__()):
            trainWordsFrequency[i,:] = self.dataset.toFrequencies(self.dataset.trainTexts[i], self.dataset.bestWords)
        # Calculate Prior Probabilities
        for class_index in range(0,self.dataset.labels.__len__()):
            self.p_classes[class_index] = self.dataset.trainLabelsNum[np.where(self.dataset.trainLabelsNum==class_index)].__len__()/self.dataset.trainLabelsNum.__len__()
        # Calculate Conditional Probabilities P(each Word|each Class)
        for class_index in range(0, self.dataset.labels.__len__()):
            class_data_indexs = np.where(self.dataset.trainLabelsNum == class_index)
            for word_index in range(0,self.dataset.numBestWords):
                T_c = np.sum(trainWordsFrequency[class_data_indexs,word_index])
                self.p_words_classes[word_index,class_index] = T_c+1
            self.p_words_classes[:, class_index] = self.p_words_classes[:, class_index] / np.sum(
                    self.p_words_classes[:, class_index])


    def predict_unigram(self, testTexts) -> np.array:
        testWordsFrequency = np.zeros((testTexts.__len__(), self.dataset.numBestWords))
        for i in range(0,testTexts.__len__()):
            testWordsFrequency[i,:] = self.dataset.toFrequencies(testTexts[i], self.dataset.bestWords)
        numTestData = testWordsFrequency.shape[0]
        testLabelsPredict = np.zeros(numTestData)
        for i in range(0,numTestData):
            wordFrequencies = testWordsFrequency[i,:]
            # scores by sum of logs
            scores = np.log10(self.p_classes)
            for class_index in range(0, self.dataset.labels.__len__()):
                for word_index in range(0, self.dataset.numBestWords):
                    scores[class_index] = scores[class_index] + wordFrequencies[word_index]*np.log10(self.p_words_classes[word_index,class_index])
            # Classify by argmax Maximum Posteriory
            testLabelsPredict[i] = np.argmax(scores)
        return testLabelsPredict


    def train_bigram(self, discount):
        self.p_words_classes = np.zeros((self.dataset.numWordsBigram, self.dataset.labels.__len__()))
        self.discount = discount
        # Extract Word Frequencies Representation for Each Class
        classWordsFrequency = np.zeros((self.dataset.labels.__len__(),self.dataset.numWordsBigram))
        for class_index in range(0, self.dataset.labels.__len__()):
            classTexts = self.dataset.trainTexts[np.where(self.dataset.trainLabelsNum == class_index)]
            classTextPool = " ".join(np.array(classTexts))
            classWordsFrequency[class_index,:] = self.dataset.toFrequencies(classTextPool, self.dataset.allWordsBigram)
        # Calculate Prior Probabilities
        for class_index in range(0,self.dataset.labels.__len__()):
            self.p_classes[class_index] = np.where(self.dataset.trainLabelsNum==class_index).__len__()/self.dataset.trainLabelsNum.__len__()
        # Calculate Background Probabilities P(each Word|each Class) or 1/V
        for class_index in range(0, self.dataset.labels.__len__()):
            for word_index in range(0,self.dataset.numWordsBigram):
                T_c = classWordsFrequency[class_index,word_index]
                self.p_words_classes[word_index,class_index] = T_c+1
            self.p_words_classes[:, class_index] = self.p_words_classes[:, class_index] / np.sum(
                    self.p_words_classes[:, class_index])

        # Count each Bigram in each Class
        bigramCounts = np.zeros((self.dataset.labels.__len__(), self.dataset.allBigrams.__len__()))
        for i in range(self.dataset.trainTexts.__len__()):
            words = np.array(self.dataset.tokenize(self.dataset.trainTexts[i]))
            # remove stop words
            wordS_in_dic = np.isin(words, self.dataset.allWordsBigram)
            words = words[np.where(wordS_in_dic)]
            for j in range(words.__len__()):
                bigram = ""
                if(j>0):
                    bigram = words[j-1]+" "+words[j]
                else:
                    bigram = words[j]
                bigramIndex = self.dataset.allBigrams.index(bigram)
                bigramCounts[int(self.dataset.trainLabelsNum[i])][bigramIndex] = bigramCounts[int(self.dataset.trainLabelsNum[i])][bigramIndex]+1
        # Calculate Bigram Conditionals
        self.bigramConditional = np.zeros((self.dataset.labels.__len__(), self.dataset.allBigrams.__len__()))
        for i in range(self.dataset.allBigrams.__len__()):
            splitted = self.dataset.allBigrams[i].split(" ")
            firstWord = splitted[0]
            if(splitted.__len__()==2):
                secondWord = splitted[1]
            else:
                secondWord = splitted[0]
            B = np.where(bigramCounts[:,i] != 0)[0].__len__()
            countWord = self.dataset.allWordsBigramCount[self.dataset.allWordsBigram.index(firstWord)]/2
            alpha = discount*B / countWord
            for class_index in range(self.dataset.labels.__len__()):
                P_background = self.p_words_classes[self.dataset.allWordsBigram.index(secondWord),class_index]
                self.bigramConditional[class_index][i] = max(bigramCounts[class_index][i]-discount,0.0)/countWord + alpha* P_background


    def predict_bigram(self, testTexts) -> np.array:
        numTestData = testTexts.__len__()
        testLabelsPredict = np.zeros(numTestData)

        for i in range(0,testTexts.__len__()):
            words = np.array(self.dataset.tokenize(testTexts[i]))
            # remove stop words
            wordS_in_dic = np.isin(words, self.dataset.allWordsBigram)
            words = words[np.where(wordS_in_dic)]
            # scores by sum of logs
            scores = np.log10(self.p_classes)
            for j in range(words.__len__()):
                bigram = ""
                if (j > 0):
                    bigram = words[j - 1] + " " + words[j]
                else:
                    bigram = words[j]
                if(self.dataset.allBigrams.__contains__(bigram)):
                    bigramIndex = self.dataset.allBigrams.index(bigram)
                    for class_index in range(0, self.dataset.labels.__len__()):
                        scores[class_index] = scores[class_index] + np.log10(self.bigramConditional[class_index,bigramIndex])
                else:
                    if (self.dataset.allWordsBigram.__contains__(words[j])):
                        word_index = self.dataset.allWordsBigram.index(words[j])
                        for class_index in range(0, self.dataset.labels.__len__()):
                            scores[class_index] = scores[class_index] + np.log10(self.p_words_classes[word_index, class_index])

            # Classify by argmax Maximum Posteriory
            testLabelsPredict[i] = np.argmax(scores)
        return testLabelsPredict