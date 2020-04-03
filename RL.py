
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:31:54 2017

@author: mlopes
"""
import numpy as np
import random

from tempfile import TemporaryFile
outfile = TemporaryFile()
class finiteMDP:

    def __init__(self, nS, nA, gamma, P=[], R=[], absorv=[], alpha = 0.1):
        self.nS = nS
        self.nA = nA
        self.gamma = gamma
        self.Q = np.zeros((self.nS,self.nA))
        self.P = P
        self.R = R
        self.absorv = absorv
        self.alpha = alpha

    def runPolicy(self, n, x0,  poltype = 'greedy', polpar=[]):
        #nao alterar
        traj = np.zeros((n,4))
        x = x0
        J = 0
        for ii in range(0,n):
            a = self.policy(x,poltype,polpar)
            r = self.R[x,a]
            y = np.nonzero(np.random.multinomial( 1, self.P[x,a,:]))[0][0]
            traj[ii,:] = np.array([x, a, y, r])
            J = J + r * self.gamma**ii
            if self.absorv[x]:
                y = x0
            x = y
        return J,traj


    def VI(self):
        #nao alterar
        nQ = np.zeros((self.nS,self.nA))
        while True:
            self.V = np.max(self.Q,axis=1)
            for a in range(0,self.nA):
                nQ[:,a] = self.R[:,a] + self.gamma * np.dot(self.P[:,a,:],self.V)
            err = np.linalg.norm(self.Q-nQ)
            self.Q = np.copy(nQ)
            if err<1e-7:
                break

        #update policy
        self.V = np.max(self.Q,axis=1)
        #correct for 2 equal actions
        self.Pol = np.argmax(self.Q, axis=1)

        return self.Q,  self.Q2pol(self.Q)


    def traces2Q(self, trace):

        for i in range(0, int(1/self.alpha)):
            for entry in trace:
                #calcular valor do Q para a dada entrada
                reward = entry[3]
                q_atual = self.Q[int(entry[0]), int(entry[1])]
                max_q = max(self.Q[int(entry[2])])
                q_atualizado = q_atual + self.alpha * (reward + self.gamma * max_q - q_atual)
                self.Q[int(entry[0]), int(entry[1])] = q_atualizado
        return self.Q

    def policy(self, x, poltype = 'exploration', par = []):

        if poltype == 'exploitation':
            #indice maximo
            a = np.argmax(par[x], axis=0)


        elif poltype == 'exploration':
            a = random.randint(0, self.nA - 1)



        return a

    def Q2pol(self, Q, eta=5):
        # implementar esta funcao
        return np.exp(eta*Q)/np.dot(np.exp(eta*Q),np.array([[1,1],[1,1]]))
