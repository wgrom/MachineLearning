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
# Preparing the Data and Plot
# ================================================== #

def GeneratingData():
    # set seed for testing:
    #numpy.random.seed(100)
     
    ## Creating Input
    global classA, classB, inputs,targets
    classA = numpy.concatenate( 
            (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
             numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5])) 
    classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , 0.25]
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
 
 

# ================================================== #
# Initialization of the Program
# ================================================== #
#Start values for alfa vector
#N - number of training samples, each has corresp. alfa


## Functions

def kernel(x,y,c): 
    if c == 1: # linear
        return numpy.dot(x,y)
    elif c == 2: # polynomial
        p=3
        return math.pow(numpy.dot(x,y)+1,p)
    elif c == 3: #radial
        param = 0.5
        explicitEuclidian=numpy.linalg.norm((x[0]-y[0],x[1]-y[1]))
        return math.exp(-math.pow(explicitEuclidian,2)/(2*math.pow(param,2)))

def FormingP(t,x):
    global P
    P =numpy.zeros((N,N))    # Initialization of Matrix
    for i in range(N):
        for j in range(N):
            P[i,j] = t[i]*t[j]*kernel(x[i],x[j],c)
    return P
    
def objective(alpha):
    v = 0
    for i in  range(N):
        for j in range(N):
            v += 1/2*numpy.dot(alpha[i],alpha[j])*P[i,j]
    return v - numpy.sum(alpha)

#def objective(alpha):
#    v = 1/2 * numpy.sum( numpy.dot(alpha , alpha) * P )
#    return v - numpy.sum(alpha)

def zerofun(alpha):
    v = 0
    v = numpy.sum([alpha[a]*targets[a] for a in range(N)])
    return v

threshold = 0.00001
def extractSupportVectors(testdata, targetdata, alphaValues):
    supportVectors = []
    for i in range(len(testdata)):
        if abs(alphaValues[i]) > threshold:
            supportVectors.append((alphaValues[i], targetdata[i], testdata[i]))
    return supportVectors

def indicator(xVec, yVec, supportVec):
    result =0
    result = numpy.sum([ supportVec[i][0]*supportVec[i][1]*kernel(supportVec[i][2],(xVec, yVec),c) 
                    for i in range(len(supportVec))])
    bScalar = threshold_value(supportVec)
    result = result - bScalar
    return result

def threshold_value(supportVectors):
    b = 0
    b = numpy.sum([ supportVectors[i][0]*supportVectors[i][1]*kernel(supportVectors[0][2],supportVectors[i][2],c) 
                    for i in range(len(supportVectors))]) - supportVectors[0][1]
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
    plt.xlim((-2,2))
    plt.ylim((-2,2))
    #plt.savefig(’svmplot.pdf’) # Save a copy in a file 
    plt.show() # Show the plot on the screen 

def dispBoundaries(supportVec):
    xgrid=numpy.linspace(-5, 5) 
    ygrid=numpy.linspace(-4, 4)
    grid = numpy.array([[indicator(x,y,supportVec) for x in xgrid ] for y in ygrid]) # here I changed the way they create the matrix
                
    plt.figure(figsize=(12,6))
    plt.contour (xgrid , ygrid , grid , (-1.0, 0.0, 1.0),
             colors=('red', 'black', 'blue'), linewidths=(1, 5, 1))
    plt.plot([p[0] for p in classA ] ,
             [p[1] for p in classA], 
             'b.') 
    plt.plot([p[0] for p in classB ] ,
             [p[1] for p in classB], 
             'r.')
    plt.plot([p[2][0] for p in supportVec],
             [p[2][1] for p in supportVec],'o','m',markersize=8)

    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.show()


# ================================================== #
# Main Program
# ================================================== #    
def run()
    global c
    c = 3
        
    GeneratingData()
    start = numpy.zeros(N)
    C=0.5
    B = [(0, C) for b in range(N)]
    XC={'type': 'eq', 'fun':zerofun}
    
    
    FormingP(targets,inputs)
    
    ret = minimize( objective , start , bounds=B, constraints=XC)
    alpha = ret['x']
    
    supportVec = extractSupportVectors(inputs,targets,alpha)
    
    dispBoundaries(supportVec)
    
    
    
    print(ret['success'])




    




