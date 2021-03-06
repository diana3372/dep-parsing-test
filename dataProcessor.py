#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:28:08 2017

@author: diana
"""

from collections import defaultdict

def buildDictionaries(sentenceDependencies, unknownMarker):        
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    l2i = defaultdict(lambda: len(l2i))
    i2w = dict()
    i2t = dict()
    i2l = dict()
    
    for s in sentenceDependencies:
        for k,v in s.tokens.items():
            i2w[w2i[v.word]] = v.word
            i2t[t2i[v.POSTag]] = v.POSTag
            i2l[l2i[v.label]] = v.label
            
    w2i = defaultdict(lambda: w2i[unknownMarker], w2i)
    
    return w2i, t2i, l2i, i2w, i2t, i2l

def getTrainingSetsForWord2Vec(sentenceDependencies):        
    wordsTrainingSet = []
    posTagsTrainingSet = []
    
    for s in sentenceDependencies:
        sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags()            
        wordsTrainingSet.append(sentenceInWords[1:]) # skip root
        posTagsTrainingSet.append(sentenceInTags[1:]) # skip root
        
    return wordsTrainingSet, posTagsTrainingSet