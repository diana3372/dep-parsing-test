#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:49:31 2017

@author: diana
"""
from sentenceDependencies import SentenceDependencies, Token
from collections import Counter
import copy

commentSymbol = '#'
itemsSeparator = '\t'
undefinedField = '_'
    
class ConlluFileReader:
    def __init__(self, filePath):
        self.filePath = filePath
        self.sentencesDependencies = None
        
    def readSentencesDependencies(self, rootMarker=None):
        self.wordCounts = Counter()
        f = open(self.filePath, 'r')
        sentencesDeps = []
        sentenceDep = SentenceDependencies()
        
        for line in f.readlines():
            if line.startswith(commentSymbol):
                continue
            
            if line.isspace(): # end of the sentence
                sentencesDeps.append(sentenceDep)
                sentenceDep = SentenceDependencies()
                continue
                
            items = line.split(itemsSeparator)
            
            # this can be a float or a range if the word is implicit in the sentence
            if not float(items[0]).is_integer():
                continue
                
            index = int(items[0]) 
            head = items[6] # this can be '_' for implicit words that were added, change to -1
            if head == undefinedField:
                head = -1
            else:
                head = int(head)
            
            # Insert the root node first
            if rootMarker != None and index == 1:
                sentenceDep.addToken(Token(index=0, word=rootMarker, POSTag=rootMarker, head=0, label='root'))            
            
            sentenceDep.addToken(Token(index=index, word=items[1], POSTag=items[3], head=head, label=items[7]))
            self.wordCounts[items[1]] += 1

        f.close()
        
        self.sentencesDependencies = sentencesDeps
        return sentencesDeps
    
    def getSentenceDependenciesUnknownMarker(self, unknownMarker):
        if self.sentencesDependencies == None:
            sentencesDeps = copy.deepcopy(self.readSentencesDependencies())
        else:
            sentencesDeps = copy.deepcopy(self.sentencesDependencies)
        
        # Replace words with count = 1 with <unk>
        for s in sentencesDeps:
            for k,v in s.tokens.items():
                if self.wordCounts[v.word] == 1:
                    v.word = unknownMarker
        
        return sentencesDeps
        

class ConlluFileWriter:
    def __init__(self, filePath):
        self.filePath = filePath
    
    def getFormattedIndex(self, index):
        return '%d' % (index)
        
    def write(self, sentenceDependencies):
        f = open(self.filePath, 'w')
        lines = []        
        for sentenceDep in sentenceDependencies:
            sentence = []            
            itemsLines = []
            for k, v in sentenceDep.tokens.items():
                sentence.append(v.word)
                items = []
                
                items.append(self.getFormattedIndex(v.index))
                items.append(v.word)
                items.append(undefinedField)
                items.append(v.POSTag)
                items.append(undefinedField)
                items.append(undefinedField)
                if v.head == -1:
                    items.append(undefinedField)
                else:
                    items.append(self.getFormattedIndex(v.head))
                items.append(v.label)
                items.append(undefinedField)
                items.append(undefinedField)
                itemsLines.append(itemsSeparator.join(items))            
            
            lines.append("{0} text = {1}".format(commentSymbol, ' '.join(sentence)))
            lines.append("{0}\n".format('\n'.join(itemsLines)))
            
        f.write('{}\n'.format('\n'.join(lines)))
        f.close()
