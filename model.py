#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:56 2017

@author: diana
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from MLP import MLP

#def disableTrainingForEmbeddings(*embeddingLayers):
#    for e in embeddingLayers:
#        e.weight.requires_grad = False

class DependencyParseModel(nn.Module):
    def __init__(self, word_embeddings_dim, tag_embeddings_dim, vocabulary_size, tag_uniqueCount, label_uniqueCount, pretrainedWordEmbeddings=None, pretrainedTagEmbeddings=None):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocabulary_size, word_embeddings_dim)
        if pretrainedWordEmbeddings.any():
            assert pretrainedWordEmbeddings.shape == (vocabulary_size, word_embeddings_dim)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrainedWordEmbeddings))
        
        self.tag_embeddings = nn.Embedding(tag_uniqueCount, tag_embeddings_dim)
        if pretrainedTagEmbeddings.any():
            assert pretrainedTagEmbeddings.shape == (tag_uniqueCount, tag_embeddings_dim)
            self.tag_embeddings.weight.data.copy_(torch.from_numpy(pretrainedTagEmbeddings))
        
        # Save computation time by not training already trained word vectors
#        disableTrainingForEmbeddings(self.word_embeddings, self.tag_embeddings)
        # Now we need to train the embeddings for <root> and <unk>
        
        self.inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        self.hiddenSize = self.inputSize #* 2 # 512? is this the same as outputSize?
        self.nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, bidirectional=True)
        
        self.nDirections = 2
        self.batch = 1 # this is per recommendation
 
        # Input size of the MLP for arcs scores is the size of the output from previous step concatenated with another of the same size
        biLstmOutputSize = self.hiddenSize * self.nDirections
        mlpForScoresInputSize = biLstmOutputSize * 2
        self.mlpArcsScores = MLP(mlpForScoresInputSize, hidden_size=mlpForScoresInputSize, output_size=1)
        
        # MLP for labels
        self.label_uniqueCount = label_uniqueCount
        self.mlpLabels = MLP(mlpForScoresInputSize, hidden_size=mlpForScoresInputSize, output_size=self.label_uniqueCount)
        
    def initHiddenCellState(self):
        hiddenState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize), requires_grad=False)
        cellState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize), requires_grad=False)
        
        return hiddenState, cellState    
    
    def forward(self, words_tensor, tags_tensor, arcs_refdata_tensor):
        scoreVariable = self.predictArcs(words_tensor, tags_tensor)
        labelVariable = self.predictLabels(arcs_refdata_tensor)

        return scoreVariable, labelVariable

    def predictArcs(self, words_tensor, tags_tensor):        
         # BiLSTM        
        wordEmbeds = self.word_embeddings(words_tensor)
        tagEmbeds = self.tag_embeddings(tags_tensor)
        
        assert words_tensor.size()[0] == tags_tensor.size()[0]
        nWordsInSentence = words_tensor.size()[0]
        
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)
        
        self.hVector, (self.hiddenState, self.cellState) = self.biLstm(inputTensor.view(nWordsInSentence, self.batch, self.inputSize), (self.hiddenState, self.cellState))
        
        # MLP for arcs scores
        
        # Creation of dependency matrix. size: (length of sentence)  x (length of sentence)
        scoreVariable = Variable(torch.zeros(nWordsInSentence, nWordsInSentence))
        # We know root goes to root, so adding a 1 there
        scoreVariable[0,0] = 1.0

        # All possible combinations between head and dependent for the given sentence
        for head in range(nWordsInSentence):
            for dep in range(nWordsInSentence):
                if head != dep: 
                    hvectorConcatForArcs = torch.cat((self.hVector[head, :, :], self.hVector[dep, :, :]), 1)
                    scoreVariable[head, dep] = self.mlpArcsScores(hvectorConcatForArcs)
        
        return scoreVariable
    
    def predictLabels(self, headsIndices_tensor):        
        # MLP for labels
        # Creation of matrix with label-probabilities. size: (length of sentence) x (unique tag count)
        labelVariable = Variable(torch.zeros(len(headsIndices_tensor), self.label_uniqueCount))
        
        assert len(headsIndices_tensor) == self.hVector.size()[0]

        for i, head in enumerate(headsIndices_tensor):
            hvectorConcatForLabels = torch.cat((self.hVector[i, :, :], self.hVector[head, :, :]), 1)
            labelVariable[i, :] = self.mlpLabels(hvectorConcatForLabels)
        
        return labelVariable