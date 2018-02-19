#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:46:40 2018

@author: Simon
"""
import numpy , random , math
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
 
# ================================================== #
# Initialization of the Program
# ================================================== #

## Functions
global c
c = 1
def kernel(x,y,c): 
    if c == 1: # linear
        return numpy.dot(x,y)
    elif c == 2: # polynomial
        return math.pow(numpy.dot(x,y)+1,2)
    elif c == 3: #radial
        param = 0.01
        return math.exp**(math.pow(numpy.linalg.norm(x-y),2)/(2*math.pow(param,2)))

def FormingP(x,y):
    global P
    P =numpy.zeros((N,N))    # Initialization of Matrix
    for a in range(N):
        for b in range(N):
            P[a,b] = x[a]*x[b]*kernel(y[a],y[b],c) #c is for kernel selection
    
def objective(alpha):
    v = numpy.zeros(N,N)
    for a in  range(N):
        for b in range(N):
            v[a,b]= 1/2*numpy.dot(alpha[a],alpha[b])*P[a,b]
    return numpy.sum(v)-numpy.sum(alpha)
    
def zerofun(alpha,t):
    for a in range(N):
        v = numpy.dot(alpha[a],t[a])
    return numpy.sum(v)

threshold = 0.00001
def extractSupportVectors(testdata, targetdata, alphaValues):
    supportVectors = []
    for i in range(len(testdata)):
        if abs(alphaValues[i]) > threshold:
            supportVectors.append((alphaValues[i], targetdata[i], testdata[i]))
    return supportVectors

def indicator(xVec, supportVec):
    result =0
    for a in range(len(supportVec)):
        alphaData, targetData, testData = supportVec[a]
        bScalar = threshold_value(supportVec,xVec)
        result += alphaData * targetData*kernel(testData,xVec)
    result = result - bScalar
    return result

def threshold_value(supportVectors,x):
    b=[numpy.sum(supportVectors[i][0]*supportVectors[i][1]*kernel(supportVectors[i][2],x[i],c)-supportVectors[i][1]) 
    for i in range(len(supportVectors))]
    return b

# plotting:
def disp(ClassA,ClassB):
    plt.plot([p[0] for p in classA ] ,
             [p[1] for p in classA], 
             'b.') 
    plt.plot([p[0] for p in classB ] ,
             [p[1] for p in classB], 
             'r.')
    plt.axis('equal') # Force same scale on both axes 
    #plt.savefig(’svmplot.pdf’) # Save a copy in a file 
    plt.show() # Show the plot on the screen 

# ================================================== #
# Preparing the Data
# ================================================== #

# set seed for testing:
numpy.random.seed(100) 
## Creating Inputs
classA = numpy.concatenate( 
        (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
         numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5])) 
classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]
 
inputs = numpy.concatenate (( classA , classB )) 

targets = numpy.concatenate (
        (numpy.ones(classA.shape[0]) , 
         -numpy.ones(classB.shape[0])))
 
# Number of rows (samples)
global N 
N = inputs.shape[0] 
 
# reorder the samples ramdomly
permute= list(range(N)) 
random.shuffle(permute) 
inputs = inputs[ permute , : ]
targets = targets[ permute ]

#Start values for alfa vector
#N - number of training samples, each has corresp. alfa
start = numpy.zeros(N)
 
# list of pairs as long as alfa
C=1
bounds = [(0, C) for b in range(N)]
# XC equality constraint of (10)
#constraint={'type': eq, 'fun':zerofun}


# ================================================== #
# Main Program
# ================================================== #   

#def minimize( objective , start , bounds=B, constraints=XC):
#    alpha = ret['x']    
#    solution = ret['success']




xgrid=numpy.linspace(-5, 5) 
ygrid=numpy.linspace(-4, 4)
grid=numpy.array([[indicator(x, y) for y in ygrid ] for x in xgrid])
plt.contour (xgrid , ygrid , grid , (-1.0, 0.0, 1.0),
             colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    




