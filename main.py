
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
alpha = 0.4  # try 0.05 for e-greedy, 0.5 for softmax
lamb = 0.4  # conservative due to accumulating traces
xt = numpy.array(numpy.zeros((vsize, vsize)))
xtp1 = numpy.array(numpy.zeros((vsize, vsize)))
R = 0  # Reward
Ravg = 0  # Average Reward
angle1 = angle2 = 0  # Observation/state signals
action1 = action2 = 1  # Action
action1new = action2new = 1  # Action
mean1 = mean2 = 0  # Policy params (means) [Cont. ACRL.py]
sigma1 = sigma2 = 1  # Policy params (sigmas) [Cont. ACRL.py]
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
control1 = ACRL.ACRL(continuous=False, gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)
control2 = ACRL.ACRL(continuous=False, gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)#Sarsa.Sarsa(gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)


def getAction(action, angle):
    if action == 0:
        return 0.0
    elif action == 1:
        return -0.05
    elif action == 2:
        return 0.05
    elif action == 3:
        return -0.1
    elif action == 4:
        return 0.1



# === Main Loop ===
print("Starting Main Loop ...")
s1.move_angle(0)
s2.move_angle(0)
for i in range(0, 10000):
    start = time.time()

    # ==== States and actions resolve here ====


    # Apply action via angular control command
 #   print 'action 2 is: ' + str(action2)
    #action1proc = getAction(action1, angle1)
    #action2proc = getAction(action2, angle2)
    print action1
    action1proc = (action1 - 0.5) * 0.1  # generate an angular change between -0.05 and 0.05
    action2proc = (action2 - 0.5) * 0.1  # generate an angular change between -0.05 and 0.05

    newangle1 = safety_check(action1proc + angle1)
    newangle2 = safety_check(action2proc + angle2)
   # s1.move_angle(newangle1, blocking=True)     # to make it harder to learn, blocking = False!
    s2.move_angle(newangle2, blocking=True)

    # Observe next state
    angle1 = s1.read_angle()
    angle2 = s2.read_angle()

    # Set signal vector
    sigs_t = numpy.zeros(numsigs)
    sigs_t[0] = angle1
    sigs_t[1] = angle2
    sigs_t[2] = action1proc
    sigs_t[3] = action2proc
    sigs_t[4] = 0
    sigs_t[5] = 0

    # Get reward
    R = -abs(0.5 - angle2)# - abs(0.0-angle1)

    # Turn observation into a 2D tabular state feature vector   CRAPPY TILECODER
    xtp1 = numpy.zeros((vsize, vsize))
    xtp1[int((angle1+0.51) * 29), int((angle2+0.51) * 29)] = 1
    #xtp1[int((angle1 + 0.51) * 19), int((angle2 + 0.51) * 19)] = 1

    # Choose action Atp1 from policy
    #action1new, probs1 = control1.get_action_egreedy(xtp1.flatten(), 0.05)
    #action2new, probs2 = control2.get_action_egreedy(xtp1.flatten(), 0.05)
    action1new,probs1 = control1.get_action_softmax(xt.flatten())
    action2new,probs2 = control2.get_action_softmax(xt.flatten())

    # ==== Learning happens here ====

    # Update the control learner
    dstart = time.time()
    tderr1 = control1.update(xt.flatten(),xtp1.flatten(),R,action1,action1new,gamma, probs1) #0
    tderr2 = control2.update(xt.flatten(), xtp1.flatten(), R, action2, action2new, gamma, probs2)
    Ravg = control2.Ravg
    dend = time.time()

    # Fill these in when you make your continuous action ACRL.py learner!
    mean1 = 0.0
    mean2 = 0.5
    sigma1 = 0.2
    sigma2 = 0.4
    plotData = {'mean1': mean1, 'mean2': mean2, 'sigma1': sigma1, 'sigma2': sigma2, 'tderr1': tderr1, 'tderr2': tderr2, 'R': R,'Ravg': Ravg, 'probs1': probs1, 'probs2': probs2, 'xtp1': xtp1, 'sigs_t': sigs_t}
    plotter.plotUpdate(plotData)

    xt = xtp1
    action1 = action1new
    action2 = action2new

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
        print('Saving Figure')
        plotter.saveFigure()
