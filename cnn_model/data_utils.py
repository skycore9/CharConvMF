#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import gzip

class Data(object):
    def __init__(self, data_source, alphabet, no_of_classes, l0 = 1014, is_polarity=False):
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}
        self.is_polarity = is_polarity
        for i, c in enumerate(self.alphabet):
            self.dict[c] = i + 1
        
        self.length = l0
        self.data_source = data_source
        self.no_of_classes = no_of_classes

    def _parse(self):
        g = gzip.open(self.data_source, 'r')
        for l in g:
            yield eval(l)

    def loadData(self):
        data = []
        for line in self._parse():
            rating = int(line['overall'])
            reviewText = line['reviewText']
            if len(reviewText) >= 100:
                if self.is_polarity:
                    if rating == 3:
                        continue
                    elif rating == 1 or rating == 2:
                        rating = 1
                    else:
                        rating = 2
                data.append((rating, reviewText))
        self.data = data

    def getAllData(self):
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in self.data:
            batch_indices.append(self.strToIndexs(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def strToIndexs(self, s):
        s = s.lower()
        m = len(s)
        n = min(m, self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in xrange(1, n+1):
            c = s[-i]
            if c in self.dict:
                str2idx[i-1] = self.dict[c]
        return str2idx