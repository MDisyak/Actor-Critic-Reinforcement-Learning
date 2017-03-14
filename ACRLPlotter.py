import myLib
import matplotlib.pyplot as plt
import datetime
import time
import csv
import threading
import numpy
from pylab import *


class ACRLPlotter:
    # === Create the plotting buffers ===
    numactors = 0  # number of control learners
    numactions = 0  # number of discrete actions (if applicable)
    numsigs = 0  # numbers of other signals to be plotted
    bufflen = 100  # length of the signal buffer
    vbufflen = 100  # length of the reward/error buffers
    vsize = 0  # vector size (number of bins) for the state in each dimension


    def __init__(self, numactions, numactors, numsigs, vsize):

        self.numactions = numactions
        self.numactors = numactors
        self.numsigs = numsigs
        self.vsize = vsize

        self.sigs_t = numpy.zeros(numsigs)  # Stores current signals
        self.buff_probs = numpy.zeros((self.bufflen, numactors * numactions))
        self.buff_means = numpy.zeros((self.bufflen, numactors))
        self.buff_sigmas = numpy.zeros((self.bufflen, numactors))
        self.buff_sigs = numpy.zeros((self.bufflen, numsigs))
        self.buff_tderrs = numpy.zeros((self.vbufflen, numactors))
        self.buff_rewards = numpy.zeros((self.vbufflen, 1))
        self.buff_avgrewards = numpy.zeros((self.vbufflen, 1))
        self.li1sigs = []  # Stores lines from the signals plot
        numactions = 0  # number of discrete actions (if applicable)
        numsigs = 0  # numbers of other signals to be plotted
        bufflen = 100  # length of the signal buffer
        self.vbufflen = 100  # length of the reward/error buffers
        sigs_t = numpy.zeros(numsigs)  # Stores current signals
        buff_probs = numpy.zeros((bufflen, numactors * numactions))
        buff_means = numpy.zeros((bufflen, numactors))
        buff_sigmas = numpy.zeros((bufflen, numactors))
        buff_sigs = numpy.zeros((bufflen, numsigs))
        buff_tderrs = numpy.zeros((self.vbufflen, numactors))
        buff_rewards = numpy.zeros((self.vbufflen, 1))
        buff_avgrewards = numpy.zeros((self.vbufflen, 1))
        li1sigs = []  # Stores lines from the signals plot

        #Initialize plot
        self.initPlot()

        # === Set up the plot ===
    def initPlot(self):
        self.fig = plt.figure(figsize=(14, 8), dpi=100, )
        self.ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3)
        plt.title("Actions ($A1_t$, green, $A2_t$, red) and Signals ($\\theta1_t$ and $\\theta2_t$, dotted green and red) ",
                  fontsize=14, fontweight='bold')
        plt.ylabel("$O_t$ and $A_t$")
        self.ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=3)
        plt.ylabel("$\mu_{t+1}$ (green and red)\n$\sigma_{t+1}$ (dotted)")
        self.ax2 = plt.subplot2grid((3, 4), (2, 0), colspan=3)
        plt.ylabel("$Reward_t$ (blue)\n$AvgReward_t$ (dashed)")
        plt.xlabel("Time steps")
        self.ax4 = plt.subplot2grid((3, 4), (0, 3))
        plt.title("$x_{t+1}$", fontsize=14, fontweight='bold')
        self.ax4.set_yticklabels([])
        self.ax4.set_xticklabels([])
        self.ax5 = plt.subplot2grid((3, 4), (1, 3))
        plt.ylabel("Action Values / Probs")
        self.ax5.yaxis.tick_right()
        self.ax6 = plt.subplot2grid((3, 4), (2, 3))
        self.ax6.yaxis.tick_right()
        plt.ylabel("TD Error")


        # === Create data structures to update the plot data ===
        self.li1sigs = self.ax1.plot(self.buff_sigs)
        # Has for clarity; assumes exactly 6 signals as defined for this experiment
        plt.setp(self.li1sigs[0], linewidth=2, color='#77DD77', drawstyle="steps-post", alpha=0.8, dash_capstyle="round",
             linestyle=":")
        plt.setp(self.li1sigs[1], linewidth=2, color='#EE7777', drawstyle="steps-post", alpha=0.8, dash_capstyle="round",
             linestyle=":")
        plt.setp(self.li1sigs[2], linewidth=2, color='#009900', drawstyle="steps-post", alpha=0.8)
        plt.setp(self.li1sigs[3], linewidth=2, color='#AA0000', drawstyle="steps-post", alpha=0.8)
        plt.setp(self.li1sigs[4], linewidth=2, color='#AAAAAA', drawstyle="steps-post", alpha=0.05)
        plt.setp(self.li1sigs[5], linewidth=2, color='#DDDDDD', drawstyle="steps-post", alpha=0.05)

        self.li2rewards = self.ax2.plot(self.buff_rewards)
        self.li2avgrewards = self.ax2.plot(self.buff_avgrewards)
        for j in range(0, len(self.li2rewards)):
            plt.setp(self.li2rewards[j], linewidth=2, color='#5555DD', drawstyle="steps-post",
                 alpha=1 - j / float(len(self.li2rewards) + 2))
        plt.setp(self.li2avgrewards, linewidth=2, color='#AAAAEE', drawstyle="steps-post", dash_capstyle="round", linestyle="--")

        self.li3means = self.ax3.plot(self.buff_means)
        self.li3sigmas = self.ax3.plot(self.buff_sigmas)
        # for j in range(0,len(li3means)):
        #	plt.setp(li3means[j], linewidth=2, color='#DD00DD', drawstyle="steps-post",alpha=1-j/float(len(li3means)+2))
        # for j in range(0,len(li3sigmas)):
        #	plt.setp(li3sigmas[j], linewidth=2, color='#00DD00', drawstyle="steps-post",alpha=1-j/float(len(li3sigmas)+2))
        # Hack for clarity below; assumes 2 actors
        plt.setp(self.li3means[0], linewidth=2, color='#00AA00', drawstyle="steps-post", alpha=0.8)
        plt.setp(self.li3means[1], linewidth=2, color='#990000', drawstyle="steps-post", alpha=0.8)
        plt.setp(self.li3sigmas[0], linewidth=2, color='#77DD77', drawstyle="steps-post", alpha=0.8, dash_capstyle="round",
             linestyle=":")
        plt.setp(self.li3sigmas[1], linewidth=2, color='#EE7777', drawstyle="steps-post", alpha=0.8, dash_capstyle="round",
             linestyle=":")

        self.li5probs = self.ax5.plot(self.buff_probs[-20:, :])
        # for j in range(0,len(li5probs)):
        #	plt.setp(li5probs[j], linewidth=2, color='#DD0000', drawstyle="steps-post",alpha=1-j/float(len(li5probs)+2))
        # Hack for clarity below; assumes two actions per actor only
        plt.setp(self.li5probs[0], linewidth=2, color='#77DD77', drawstyle="steps-post", alpha=0.8, dash_capstyle="round",
             linestyle=":")
        plt.setp(self.li5probs[1], linewidth=2, color='#00AA00', drawstyle="steps-post", alpha=0.8)
        plt.setp(self.li5probs[2], linewidth=2, color='#EE7777', drawstyle="steps-post", alpha=0.8, dash_capstyle="round",
             linestyle=":")
        plt.setp(self.li5probs[3], linewidth=2, color='#990000', drawstyle="steps-post", alpha=0.8)

        self.li6tderrs = self.ax6.plot(self.buff_tderrs)
        # for j in range(0,len(li6tderrs)):
        #	plt.setp(li6tderrs[j], linewidth=2, color='#5555DD', drawstyle="steps-post",alpha=1-j/float(len(li6tderrs)+2))
        # Hack for clarity below; assumes 2 actors
        plt.setp(self.li6tderrs[0], linewidth=2, color='#00AA00', drawstyle="steps-post", alpha=0.8)
        plt.setp(self.li6tderrs[1], linewidth=2, color='#990000', drawstyle="steps-post", alpha=0.8)
    
        self.ax1.set_ylim((-0.1, 1))
        self.ax2.set_ylim((-0.1, 102))
        self.ax3.set_ylim((-0.1, 2))
        self.ax5.set_ylim((-0.1, 1))
    
        xtest = numpy.zeros((self.vsize, self.vsize))
        self.imbits = self.ax4.imshow(xtest, interpolation='nearest',
                    origin='bottom',
                    aspect='auto',  # get rid of this to have equal aspect
                    vmin=0,
                    vmax=1,
                    cmap=plt.cm.gray)
    
        self.fig.canvas.draw()

        # === Timing display ===
        self.t = figtext(0.30, 0.01, "Actors: " + str(self.numactors), fontsize=12, color="grey")
        self.t2 = figtext(0.45, 0.01, "Steptime: " + str(0.0) + " s", fontsize=12, color="grey")
        self.t3 = figtext(0.60, 0.01, "LearnerSteptime: " + str(0.0) + " s", fontsize=12, color="grey")
        self.t4 = figtext(0.13, 0.31, "MeanAvgReward100ts: " + str(0), fontsize=12, color="grey")
        self.t5 = figtext(0.12, 0.01, "Timesteps: " + str(0), fontsize=12, color="grey")

        plt.pause(0.05)
        
    def plotUpdate(self, plotData):
        # ==== Plotting upkeep follows ====
        self.mean1 = plotData['mean1']
        self.mean2 = plotData['mean2']
        self.sigma1 = plotData['sigma1']
        self.sigma2 = plotData['sigma2']
        self.tderr1 = plotData['tderr1']
        self.tderr2 = plotData['tderr2']
        self.R = plotData['R']
        self.Ravg = plotData['Ravg']
        self.probs1 = plotData['probs1']
        self.probs2 = plotData['probs2']
        self.xtp1 = plotData['xtp1']
        self.sigs_t = plotData['sigs_t']


        # Vectors
        means_t = numpy.array([self.mean1, self.mean2])
        sigmas_t = numpy.array([self.sigma1, self.sigma2])
        tderr_t = numpy.array([self.tderr1, self.tderr2])
        rewards_t = numpy.array(self.R)
        avgrewards_t = numpy.array(self.Ravg)
        probs_t = numpy.hstack((self.probs1, self.probs2))

        # Update the plotting buffers for the predictor
        self.buff_sigs = numpy.vstack((self.buff_sigs, self.sigs_t.reshape(1, self.numsigs)))[1:, :]

        # Update the plotting buffers for the cumulants and signals
        self.buff_means = numpy.vstack((self.buff_means, means_t.reshape(1, self.numactors)))[1:, :]
        self.buff_sigmas = numpy.vstack((self.buff_sigmas, sigmas_t.reshape(1, self.numactors)))[1:, :]

        # Update the plotting buffers for the verifier
        self.buff_rewards = numpy.vstack((self.buff_rewards, rewards_t.reshape(1, 1)))[1:, :]
        self.buff_avgrewards = numpy.vstack((self.buff_avgrewards, avgrewards_t.reshape(1, 1)))[1:, :]
        self.buff_probs = numpy.vstack((self.buff_probs, probs_t.reshape(1, self.numactors * self.numactions)))[1:, :]
        self.buff_tderrs = numpy.vstack((self.buff_tderrs, tderr_t.reshape(1, self.numactors)))[1:, :]

        # Update the prediction plots in real time
        for k in range(0, len(self.li1sigs)):
            self.li1sigs[k].set_ydata(self.buff_sigs[:, k])
        self.ax1.set_ylim((self.buff_sigs.min() - 0.1, self.buff_sigs.max() + 0.1))

        for k in range(0, len(self.li3means)):
            self.li3means[k].set_ydata(self.buff_means[:, k])
        for k in range(0, len(self.li3sigmas)):
            self.li3sigmas[k].set_ydata(self.buff_sigmas[:, k])
        self.ax3.set_ylim((self.buff_means.min() - 0.1, self.buff_means.max() + 0.1))

        # Plot REWARD
        for k in range(0, len(self.li2rewards)):
            self.li2rewards[k].set_ydata(self.buff_rewards[:, k])
        for k in range(0, len(self.li2avgrewards)):
            self.li2avgrewards[k].set_ydata(self.buff_avgrewards[:, k])
        self.ax2.set_ylim((self.buff_rewards.min() - 0.5, self.buff_rewards.max() + 0.5))

        # Plot PROBS / QVALUES
        for k in range(0, len(self.li5probs)):
            self.li5probs[k].set_ydata(self.buff_probs[-20:, k])
        self.ax5.set_ylim((self.buff_probs[-20:, :].min() - 0.1, self.buff_probs[-20:, :].max() + 0.1))

        # Plot TD Error
        for k in range(0, len(self.li6tderrs)):
            self.li6tderrs[k].set_ydata(self.buff_tderrs[:, k])
            self.ax6.set_ylim((self.buff_tderrs.min() - 0.1, self.buff_tderrs.max() + 0.1))

        # Plot Feature Vector
        self.imbits.set_data(self.xtp1)

        self.fig.canvas.draw()
        plt.pause(0.05)


    def saveFigure(self, fileName='ACRLFigure_from_%s.png' % datetime.datetime.now()):
        plt.savefig('figures/%s' % fileName)