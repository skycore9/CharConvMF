import os
import sys
from keras.models import Model, load_model
import numpy as np
import random

class Data(object):
    def __init__(self, rating_path, document_path, is_polarity):
        # the items that one user rating
        self.urs = dict()

        self.is_polarity = is_polarity

        self.avg_rating = 0.0

        # read data
        self._read_data(rating_path=rating_path, document_path=document_path)


    def _read_data(self, rating_path, document_path, min_rating=1, max_length=1014):
        if os.path.isfile(document_path):
            raw_content = open(document_path, 'r')
        else:
            print "document_path is wrong!"
            sys.exit()

        if os.path.isfile(rating_path):
            raw_ratings = open(rating_path, 'r')
        else:
            print "rating_path is wrong!"
            sys.exit()

        print "Preprocessing the rating data...."
        # 1st scan document file to filter items which have documents
        tmp_id_plot = set()
        for line in raw_content:
            tmp = line.strip().split('::')
            i = tmp[0]
            # if len(tmp[1]) <= 100:
            #     continue
            tmp_id_plot.add(i)
        raw_content.close()

        # 1st can rating file to check #ratings of each user
        tmp_user = dict()
        for line in raw_ratings:
            tmp = line.strip().split('::')
            u = tmp[0]
            i = tmp[1]
            if i in tmp_id_plot:
                if u not in tmp_user:
                    tmp_user[u] = 1
                else:
                    tmp_user[u] += 1
        raw_ratings.close()

        # 2nd scan rating file to make matrix indices of users and items
        raw_ratings = open(rating_path, 'r')
        userset = dict()
        itemset = dict()
        user_idx = 0
        item_idx = 0

        data = set()
        for line in raw_ratings:
            tmp = line.strip().split('::')
            u = tmp[0]
            if u not in tmp_user:
                continue
            i = tmp[1]

            if tmp_user[u] >= min_rating:
                if u not in userset:
                    userset[u] = user_idx
                    user_idx += 1
                if (i not in itemset) and (i in tmp_id_plot):
                    itemset[i] = item_idx
                    item_idx += 1
            else:
                continue

            if u in userset and i in itemset:
                u_idx = userset[u]
                i_idx = itemset[i]

                self.avg_rating += float(tmp[2])
                data.add((u_idx, i_idx, float(tmp[2])))
        raw_ratings.close()

        # the number of users
        self.num_users = len(userset)
        # the number of items
        self.num_items = len(itemset)
        # the number of ratings
        self.num_ratings = len(data)
        # average rating
        self.avg_rating /= self.num_ratings

        print "Finish preprocessing rating data - #user: %d, #item: %d, #ratings: %d, sparisty: %f" %\
              (self.num_users, self.num_items, self.num_ratings,\
               1 - self.num_ratings / float(self.num_users * self.num_items))

        train_num = int(self.num_ratings * 0.8)
        self.train_set = random.sample(data, train_num)
        self.test_set = data - set(self.train_set)

        for u, v, r in self.train_set:
            # neighbors
            if u not in self.urs:
                self.urs[u] = set()
            self.urs[u].add(v)

        print 'Preprocessing the item document....'
        map_id2plot = dict()
        raw_content = open(document_path, 'r')
        for line in raw_content:
            tmp = line.strip().split("::")
            if tmp[0] in itemset:
                i = itemset[tmp[0]]
                tmp_plot = tmp[1].split('|')
                map_id2plot[i] = ' '.join(tmp_plot)
        raw_content.close()

        batch_indices = []
        for text in map_id2plot.itervalues():
            batch_indices.append(self.strToIndexs(text))

        # get intermediate output
        model_name = "cnn_model5.h5"
        if self.is_polarity:
            model_name = "cnn_model2.h5"
        self.cnn_model = load_model(model_name)
        layer_name = "output1"
        intermediate_layer_model = Model(input=self.cnn_model.input,
                                     output=self.cnn_model.get_layer(layer_name).output)
        # intermediate output
        self.chi = intermediate_layer_model.predict(np.asarray(batch_indices))

        print 'Finish preprocessing item document.'

    def strToIndexs(self, s, max_length=1014):
        s = s.lower()
        m = len(s)
        n = min(m, max_length)
        str2idx = np.zeros(max_length, dtype='int64')
        for i in xrange(1, n+1):
            c = s[-i]
            if c in self.dict:
                str2idx[i-1] = self.dict[c]
        return str2idx