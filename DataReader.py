
# coding:utf-8
from __future__ import unicode_literals
import numpy as np
import os
import math
from collections import Counter


class DataReader():
    '''
    This class aimed at reading the ZebRa Dataset 
    '''

    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        self.trainTexts = np.array({""})
        self.testTexts = np.array({""})
        self.trainLabels = np.array({""})    ####???????? in alaem yani chi?
        self.testLabels = np.array({""})
        self.labels = np.array({""})
        self.trainLabelsNum = np.zeros(1)
        self.testLabelsNum = np.zeros(1)
        self.stop_words = np.array({""})
        # Calculate Best Words for Unigram
        self.bestWords = np.array({""})
        self.numWordsBigram = 200
        self.numWordsProcess = 1000 # consider 1000 most frequence for fast calculations
        self.numBestWords = -1 # -1 for all words
        # list of all train words except stop words
        self.allWordsBigram = []
        self.allWordsBigramCount = np.zeros(1)
        self.allBigrams = np.zeros(1)

    def read(self):
        # Read Train Files
        path = self.datasetPath+ "HAM-Train.txt"
        trainLabels = []
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line_splited = (line.strip().split('@@@@@@@@@@'))
                label = line_splited[0]
                text = line_splited[1]
                trainTexts.append(text)
                trainLabels.append(label)
        # Read Test Files
        path = self.datasetPath + "HAM-Test.txt"
        testTexts = []
        testLabels = []
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line_splited = (line.strip().split('@@@@@@@@@@'))
                label = line_splited[0]
                text = line_splited[1]
                testTexts.append(text)
                testLabels.append(label)

        # Extract Distinct Labels
        uniqLabels = sorted(set(trainLabels))  # remove duplicate words and sort
        for label in uniqLabels:
            print(label+" : "+str(trainLabels.count(label)))  #listi az label ha ba tedade tekrareshamn tahie mikonim
        self.trainTexts = np.array(trainTexts)
        self.testTexts = np.array(testTexts)
        self.trainLabels = np.array(trainLabels)
        self.testLabels = np.array(testLabels)
        self.labels = uniqLabels
        trainLabelsNum = np.zeros((trainLabels.__len__()))
        testLabelsNum = np.zeros((testLabels.__len__()))
        for i in range(0,uniqLabels.__len__()):
            B_in_A_bool = np.isin(trainLabels, uniqLabels[i])
            trainLabelsNum[np.where(B_in_A_bool)] = i
            B_in_A_bool = np.isin(testLabels, uniqLabels[i])
            testLabelsNum[np.where(B_in_A_bool)] = i
        self.trainLabelsNum = trainLabelsNum
        self.testLabelsNum = testLabelsNum
        # Read Farsi Stop Words
        file = open("stopwords_fa.txt", "r", encoding='utf-8')
        stop_words_text = file.read()
        stop_words = stop_words_text.split()
        self.stop_words = stop_words
        # Calculate Unigrams and Bigrams and Best Words
        self.bestInformationGain(trainTexts, trainLabels)
        # Extract Word Frequencies Representation for Each Tex
        self.numBestWords = self.bestWords.__len__()

    def tokenize(self,text):
        words = text.split(" ")
        final_words = []
        for element in words:
            word = element.replace(",", "")
            word = word.replace(".", "")
            if(~word.__eq__("")):
                final_words.append(word)
        return final_words


    def bestInformationGain(self, trainTexts, trainLabels):
        allWords = []
        allLabels = []
        for i in range(trainTexts.__len__()):
            words = self.tokenize(trainTexts[i])
            allWords.extend(words)
        allWords = np.array(allWords)
        # Remove Stop Words
        wordS_in_stopWords = np.isin(allWords, self.stop_words)
        allWords = allWords[np.where(~wordS_in_stopWords)]
        allWordsC = Counter(allWords)
        allWords = allWordsC.most_common(self.numWordsProcess)
        allWords = [word for word, count in allWords]#?????????????????????????

        # Calculate Information Gains
        if(self.numBestWords!=-1):
            IG = np.zeros((allWords.__len__()))
            classWordsFrequency = np.zeros((self.labels.__len__(), allWords.__len__()))
            p_classes = np.zeros(self.labels.__len__())
            p_words_classes = np.zeros((allWords.__len__(), self.labels.__len__()))
            for class_index in range(0, self.labels.__len__()):
                classTexts = self.trainTexts[np.where(self.trainLabelsNum == class_index)]
                classTextPool = " ".join(np.array(classTexts))
                classWordsFrequency[class_index, :] = self.toFrequencies(classTextPool, allWords)
            # Calculate Conditional Probabilities P(each Class|each Word)
            for class_index in range(0, self.labels.__len__()):
                for word_index in range(0, allWords.__len__()):
                    T_c = np.sum(classWordsFrequency[class_index, word_index])
                    p_words_classes[word_index, class_index] = T_c + 1
            for word_index in range(0, allWords.__len__()):
                # Normalize Conditional Probabilities
                p_words_classes[word_index, :] = p_words_classes[word_index, :] / np.sum(
                    p_words_classes[word_index, :])

            # Calculate Prior Probabilities
            for class_index in range(0, self.labels.__len__()):
                p_classes[class_index] = self.trainLabelsNum[np.where(self.trainLabelsNum == 0)].__len__() / self.trainLabelsNum.__len__()
            E_S = np.sum(np.multiply(-p_classes, [math.log2(p) for p in p_classes]))/ math.log2(self.labels.__len__())
            for word_index in range(0, allWords.__len__()):
                p_split_word = p_words_classes[word_index,:]#p class ha be sharte word
                E_Word = np.sum(np.multiply(-p_split_word, [math.log2(p) for p in p_split_word]))/ math.log2(self.labels.__len__())
                IG[word_index] = (E_S - E_Word)
            bestWordsIndex = np.argsort(-IG, axis=0 )

            bestWords = []
            print("Best Features are:")
            for word_index in range(0, self.numBestWords):
                word = allWords[bestWordsIndex[word_index]]
                bestWords.append(word)
                print(word +" : IG = " + str(IG[bestWordsIndex[word_index]]))
        else:
            bestWords = allWords
            self.numBestWords = allWords.__len__()

        self.bestWords = bestWords
        allWordsBigram = allWordsC.most_common(self.numWordsBigram)
        self.allWordsBigram = [word for word, count in allWordsBigram]
        self.allWordsBigramCount = np.array([count for word, count in allWordsBigram])

    def calculateAllBigrams(self):
        # Calculate All Possible Bigrams in All  Train Texts
        allBigrams = []
        counts = []
        for i in range(self.trainTexts.__len__()):
            words = np.array(self.tokenize(self.trainTexts[i]))
            # remove stop words
            wordS_in_dic = np.isin(words, self.allWordsBigram)
            words = words[np.where(wordS_in_dic)]
            for j in range(words.__len__()):
                bigram = words[j]
                if(j>0):
                    bigram = words[j-1]+" "+words[j]
                if(allBigrams.__contains__(bigram)):
                    index = allBigrams.index(bigram)
                    counts[index] = counts[index] + 1
                else:
                    allBigrams.append(bigram)
                    counts.append(1)
        self.allBigrams = (allBigrams)


    def toFrequencies(self, text, dict):
        words = self.tokenize(text)
        wordfreq = [words.count(p) for p in dict]
        return wordfreq

