import numpy 
import time
import math
import random

class ACRL:
    def __init__(self, continuous, gamma=1.0, lamb=0.4, alpha=0.1, size=10, numactions=2):
        self.gamma = 1.0
        self.alpha = alpha#alpha/10#0.05#alpha#alpha/10
        self.alpha_r = 0.01 * alpha#0.0005#0.01 * alpha
        self.alpha_mu = 0.005
        self.alpha_sigma = 0.005#alpha / 10
        self.beta = alpha #0.05#alpha#10 * alpha
        self.lamb = 0.4
        self.currentState = None
        self.nextState = None
        self.w = None
        self.e_w = None
        self.e_theta = None
        self.e_mu = None
        self.theta = None
        self.theta_sigma = None
        self.Ravg = 0
        if continuous:
            self.resetContinuous(size)
        else:
            self.reset(size, numactions)
        self.numactions = numactions
        print('size of currentState : ' + str(len(self.currentState)))
        self.mean = 0.0
        self.sigma = 0.0


    def get_action_softmax(self, x):
        v = numpy.zeros(self.numactions)
        p = numpy.zeros(self.numactions)
        # Compute values and probabilities for each action
        for i in range(0, self.numactions):
            v[i] = numpy.exp(numpy.dot(x, (self.w[:, i]).flatten()))
        for i in range(0, self.numactions):
            p[i] = v[i] / v.sum()
            # Wheel of fortune for action selection
        prob = random.random()
        psum = 0
        for i in range(0, self.numactions):
            if prob < p[i] + psum:
                a = i
                break
            else:
                psum += p[i]
        return a, p

    def reset(self, size, numactions):
        self.currentState = numpy.array(numpy.zeros((size, numactions)))
        self.nextState = numpy.array(numpy.zeros((size, numactions)))
        self.w = numpy.array(numpy.zeros((size, numactions)))
        self.e_w = numpy.array(numpy.zeros((size, numactions)))
        self.e_theta = numpy.array(numpy.zeros((size, numactions)))
        self.e_sigma = numpy.array(numpy.zeros((size, numactions)))
        self.e_mu = numpy.array(numpy.zeros((size, numactions)))
        self.theta = numpy.array(numpy.zeros((size, numactions)))

    def resetContinuous(self, size):
        self.currentState = numpy.array(numpy.zeros(size))
        self.nextState = numpy.array(numpy.zeros(size))
        self.w = numpy.array(numpy.zeros(size))
        self.e_w = numpy.array(numpy.zeros(size))
        self.e_theta = numpy.array(numpy.zeros(size))
        self.e_sigma = numpy.array(numpy.zeros(size))
        self.e_mu = numpy.array(numpy.zeros(size))
        self.theta = numpy.array(numpy.zeros(size))
        self.theta_sigma = numpy.array(numpy.zeros(size))
        
    def update(self, currentState, nextState, Rtp1, At, Atp1, gamma, probs):
        self.probs = probs
        self.gamma = gamma
        self.currentState = self.currentState * 0
        self.currentState[:, At] = currentState
        self.nextState = self.nextState * 0
        self.nextState[:, Atp1] = nextState
        self.R = Rtp1
        #Critic
        self.delta = self.R - self.Ravg + self.gamma * numpy.dot(self.nextState.flatten(), self.w.flatten()) - numpy.dot(
            self.currentState.flatten(), self.w.flatten())
        self.Ravg = self.Ravg + self.delta * self.alpha_r
        self.e_w = self.lamb * self.e_w + self.currentState
        self.w = self.w + self.beta * self.delta * self.e_w
        #Actor
        self.e_theta = self.lamb * self.e_theta + self.getDiscreteGradient(At)  # x(s,a_Taken) - prob(taking action 1)*x(s,a1) - prob(a2)*x(s,a2)#GRADIENT OF LOG (POLICY)
        self.theta = self.theta + self.alpha * self.delta * self.e_theta
        return self.delta

    def getDiscreteGradient(self, At):
        stateOut = self.currentState
        for i in range(self.numactions):
            stateOut[:,i] = stateOut[:,i] - numpy.inner(stateOut[:,At], self.probs[i])
        return stateOut

    def updateContinuous(self, currentState, nextState, Rtp1, At, gamma):
        self.gamma = gamma
        self.currentState = currentState
        self.nextState = nextState
        self.R = Rtp1
        #Critic
        self.delta = self.R - self.Ravg + self.gamma * numpy.dot(self.nextState, self.w) - numpy.dot(
            self.currentState, self.w)
        self.Ravg = self.Ravg + self.delta * self.alpha_r
        self.e_w = self.lamb * self.e_w + self.currentState
        self.w = self.w + self.beta * self.delta * self.e_w
        #Actor
        self.e_mu = self.lamb * self.e_mu + ((At - self.mean) * self.currentState)
        self.theta = self.theta + self.alpha_mu * self.delta * self.e_mu
        self.e_sigma = self.e_sigma + ((At - self.mean)**2 - self.sigma**2) * self.currentState
        self.theta_sigma = self.theta_sigma + self.alpha_sigma * self.delta * self.e_sigma
        return self.delta


    def getActionContinuous(self):
        self.mean = round(numpy.dot(self.theta, self.currentState),8) + 0.00000001
        self.sigma = round(numpy.exp(numpy.dot(self.theta_sigma,self.currentState)),8) + 0.00000001
        if self.sigma > 1.0:
            self.sigma = 1.0
        print 'mean is : ' + str(self.mean)
        print 'sigma is: ' + str(self.sigma)
        return round(numpy.random.normal(self.mean,self.sigma),4)

    def getMean(self):
        return self.mean
    def getSigma(self):
        return self.sigma