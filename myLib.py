#Author: Michael Disyak
#This file is used to store a library of functions that I deem useful


import math

def radToDeg(a):
    return a*180/math.pi
def degToRad(a):
    return a*math.pi/180
def normalizeLoad(value):
    return (value-(-1023.0))/(1023.0-(-1023.0))
def normalizeEncode(value):
    return value/1023
def normalizeAngle(value):
    return(value-(-100.0))/(100.0-(-100.0))
def normalizeDeltaAngle(value):
    return (value-(-20.0))/(20.0-(-20.0))
