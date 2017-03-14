
import myLib as myLib
import threading
import random

random.seed(0)
import GVF
import RLtoolkit.tiles as tile
import numpy
import copy
import Horde
import datetime
import ObservationManager
import signal
import termios, fcntl, sys, os
import ACRLPlotter
import time
import types
from lib_robotis_hack import *
import Sarsa
import ACRL

# Method to protect robot joint limits (template to be extended based on your own robot)
def safety_check(a):
    if a < -0.5:
        return -0.5
    if a > 0.5:
        return 0.5
    return a


# === Init Variables and Learners ===
vsize = 30  # vector size (number of bins) for the state in each dimension
numactors = 2  # number of control learners
numactions = 2  # number of discrete actions (if applicable)
numsigs = 6  # numbers of other signals to be plotted
gamma = 1.0  # using average reward setting
alpha = 0.05  # try 0.05 for e-greedy, 0.5 for softmax
lamb = 0.4  # conservative due to accumulating traces
xt = numpy.array(numpy.zeros((vsize, vsize)))
xtp1 = numpy.array(numpy.array((vsize, vsize)))
R = 0  # Reward
Ravg = 0  # Average Reward
angle1 = angle2 = 0  # Observation/state signals
action1 = action2 = 1  # Action
action1new = action2new = 1  # Action
mean1 = mean2 = 0  # Policy params (means) [Cont. ACRL.py]
sigma1 = sigma2 = 0  # Policy params (sigmas) [Cont. ACRL.py]
probs1 = probs2 = [0] * numactions  # Policy params (props/values) [Disc. ACRL.py/Sarsa]
tderr1 = tderr2 = 0  # TD Error values
toggle = False

D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU", baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D, s_list[0])
s2 = Robotis_Servo(D, s_list[1])

obsMan = ObservationManager.ObservationManager([s1,s2])
plotter = ACRLPlotter.ACRLPlotter(numactions, numactors, numsigs, vsize)

# Create two SARSA control learners
control1 = ACRL.ACRL(continuous=True, gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)
control2 = ACRL.ACRL(continuous=True, gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)#Sarsa.Sarsa(gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)





# === Main Loop ===
print("Starting Main Loop ...")
s1.move_angle(0)
s2.move_angle(0)
for i in range(0, 10000):
    start = time.time()

    # ==== States and actions resolve here ====
    # Choose action Atp1 from policy
    action1 = control1.getActionContinuous()
    action2 = control2.getActionContinuous()

    # Apply action via angular control command

    newangle1 = safety_check(action1 + angle1)
    newangle2 = safety_check(action2 + angle2)
   # s1.move_angle(newangle1, blocking=True)     # to make it harder to learn, blocking = False!
    s2.move_angle(newangle2, blocking=True)

    # Observe next state
    angle1 = s1.read_angle()
    angle2 = s2.read_angle()

    # Set signal vector
    sigs_t = numpy.zeros(numsigs)
    sigs_t[0] = angle1
    sigs_t[1] = angle2
    sigs_t[2] = action1
    sigs_t[3] = action2
    sigs_t[4] = 0
    sigs_t[5] = 0

    # Get reward
    R = -abs(0.5 - angle2)# - abs(0.0-angle1)

    # Turn observation into a 2D tabular state feature vector   CRAPPY TILECODER
    xtp1 = numpy.zeros((vsize, vsize))
    xtp1[int((angle1+0.51) * 29), int((angle2+0.51) * 29)] = 1
    #xtp1[int((angle1 + 0.51) * 19), int((angle2 + 0.51) * 19)] = 1



    # ==== Learning happens here ====

    # Update the control learner
    dstart = time.time()
    tderr1 = control1.updateContinuous(xt.flatten(),xtp1.flatten(),R,action1,gamma) #0
    tderr2 = control2.updateContinuous(xt.flatten(), xtp1.flatten(), R, action2, gamma)
    Ravg = control2.Ravg
    dend = time.time()

    # Fill these in when you make your continuous action ACRL.py learner!
    mean1 = control1.getMean()
    mean2 = control2.getMean()
    sigma1 = control1.getSigma()
    sigma2 = control2.getSigma()
    plotData = {'mean1': mean1, 'mean2': mean2, 'sigma1': sigma1, 'sigma2': sigma2, 'tderr1': tderr1, 'tderr2': tderr2, 'R': R,'Ravg': Ravg, 'probs1': probs1, 'probs2': probs2, 'xtp1': xtp1, 'sigs_t': sigs_t}
    plotter.plotUpdate(plotData)

    xt = xtp1

    #Calculate timings (used in plotter)
    end = time.time()
    latency = end - start
    dlatency = dend - dstart

    #Set text elements in plotter
    plotter.t2.set_text("Steptime: " + str.format('{0:.3f}', latency) + " s")
    plotter.t3.set_text("LearnerSteptime: " + str.format('{0:.3f}', dlatency) + " s")
    plotter.t4.set_text("MeanAvgReward100ts: " + str.format('{0:.3f}', plotter.buff_avgrewards[-101:].mean()))
    plotter.t5.set_text('Time Step: ' + str(i))

    if i % 1000 == 0:
        plotter.saveFigure()
