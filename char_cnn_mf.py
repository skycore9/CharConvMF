#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

class CharConvCF(object):
    def __init__(self, n_factors=50, n_epochs=200, lr_all=0.001, reg_all=0.1,
                 lr_bu=None, lr_bv=None, lr_pu=None, lr_qv=None, lr_yj=None,
                 reg_bu=None, reg_bv=None, reg_pu=None, reg_qv=None, reg_yj=None,
                 is_weighted=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs

        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bv = lr_bv if lr_bv is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qv = lr_qv if lr_qv is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all

        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bv = reg_bv if reg_bv is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qv = reg_qv if reg_qv is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all

        self.is_weighted = is_weighted

    def fit(self, data):
        # user biases
        self.bu = np.random.randn(data.num_users + 1)

        # item biases
        if self.is_weighted:
            self.W = np.random.randn(data.num_items + 1, data.chi.shape[1])
        else:
            self.bv = np.random.randn(data.num_items + 1)

        # user factors
        self.P = np.random.randn(data.num_users + 1, self.n_factors)

        # item factors
        self.Q = np.random.randn(data.num_items + 1, self.n_factors)

        # item implicate factors
        self.Y = np.random.randn(data.num_items + 1, self.n_factors)


        for current_epoch in xrange(self.n_epochs):
            for u, v, r in data.train_set:
                # compute user implicite feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                Nu = data.urs[u]
                sqrt_Nu = np.sqrt(len(Nu))
                for j in Nu:
                    u_impl_fdb += self.euler(data.chi[v], data.chi[j]) * self.Y[j]
                u_impl_fdb /= sqrt_Nu

                # compute current error
                dot = np.dot(self.P[u], self.Q[v] + u_impl_fdb)
                if self.is_weighted:
                    err = r - (dot + self.bu[u] + np.dot(self.W[v], data.chi[v]) + data.avg_rating)
                else:
                    err = r - (dot + self.bu[u] + self.bv[v] + data.avg_rating)

                # update biases
                self.bu[u] += self.lr_bu * (err - self.reg_bu * self.bu[u])
                if self.is_weighted:
                    self.W[v] += self.lr_bv * (err * data.chi[v] - self.reg_bv * self.W[v])
                else:
                    self.bv[v] += self.lr_bv * (err - self.reg_bv * self.bv[v])
                # update factors
                self.P[u] += self.lr_pu * (err * (self.Q[v] + u_impl_fdb) - self.reg_pu * self.P[u])
                self.Q[v] += self.lr_qv * (err * self.P[u] - self.reg_qv * self.Q[v])
                for j in Nu:
                    self.Y[j] += self.lr_yj * \
                                 (err * self.P[u] / sqrt_Nu * self.euler(data.chi[v], data.chi[j]) \
                                  - self.reg_yj * self.Y[j])

            train_rmse, test_rmse = self.evaluate(data)
            print "Epoch %d, train rmse %f and test rmse %f" % (current_epoch, train_rmse, test_rmse)

    def evaluate(self, data):
        train_rmse = 0.0
        for u, v, r in data.train_set:
            # compute user implicite feedback
            u_impl_fdb = np.zeros(self.n_factors, np.double)
            Nu = data.urs[u]
            sqrt_Nu = np.sqrt(len(Nu))
            for j in Nu:
                u_impl_fdb += self.euler(data.chi[v], data.chi[j]) * self.Y[j]
            u_impl_fdb /= sqrt_Nu

            dot = np.dot(self.P[u], self.Q[v] + u_impl_fdb)
            if self.is_weighted:
                err = r - (dot + self.bu[u] + np.dot(self.W[v], data.chi[v]) + data.avg_rating)
            else:
                err = r - (dot + self.bu[u] + self.bv[v] + data.avg_rating)
            train_rmse += err * err
        train_rmse = np.sqrt(train_rmse / len(data.train_set))

        test_rmse = 0.0
        for u, v, r in data.test_set:
            if u in data.urs:
                # compute user implicite feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                Nu = data.urs[u]
                sqrt_Nu = np.sqrt(len(Nu))
                for j in Nu:
                    u_impl_fdb += self.euler(data.chi[v], data.chi[j]) * self.Y[j]
                u_impl_fdb /= sqrt_Nu

                dot = np.dot(self.P[u], self.Q[v] + u_impl_fdb)
                if self.is_weighted:
                    err = r - (dot + self.bu[u] + np.dot(self.W[v], data.chi[v]) + data.avg_rating)
                else:
                    err = r - (dot + self.bu[u] + self.bv[v] + data.avg_rating)

                test_rmse += err * err
        test_rmse = np.sqrt(test_rmse / len(data.test_set))

        return train_rmse, test_rmse

    def cosin(self, a, b):
        num = np.dot(a, b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.5 + 0.5 * num / denom

    def euler(self, a, b):
        num = np.linalg.norm(a - b)
        return 1.0 / (1.0 + num)