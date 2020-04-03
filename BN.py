# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:51:49 2018

@author: mlopes
"""


class Node():
    def __init__(self, prob, parents = []):
        self.parents = parents
        self.prob = prob


    def computeProb(self, evid):
        if (self.parents == []):
            return [1 - self.prob[0], self.prob[0]]

        else:

            prob = self.prob

            for i in self.parents:
                prob = prob[evid[i]]

            return [1 - prob, prob]


class BN():
    def __init__(self, gra, prob):
        self.prob = prob
        self.gra = gra

    def aux(self, evid, post, value, post_value):
        flag = 0
        for i in range(0, len(evid)):
            if evid[i] == [] :
                flag = 1
                for j in range(0, 2):
                    evid_copy = evid.copy()
                    evid_copy[i] = j
                    value = self.aux(evid_copy, post, value, post_value)
                break
        if flag:
            return value
        else:
            evid[post] = post_value
            return value + self.computeJointProb(evid)


    def computePostProb(self, evid):
        post = -1
        for i in range(0, len(evid)):
            if(evid[i] == -1):
                post = i
                break

        #marginalizar
        izero = self.aux( list(evid), post, 0, 0)
        ione = self.aux( list(evid), post, 0, 1)

        #multiplicar pelo alpha
        res = (1/(izero + ione)) * ione

        return res


    def computeJointProb(self, evid):

        result = 1;
        #aplicar teorema de Bayes
        for i in range(0, len(evid)):
            result *= self.prob[i].computeProb(evid)[evid[i]]

        return result
