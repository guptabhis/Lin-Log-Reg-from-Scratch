# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:42:24 2019

@author: abhishek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import zeros
from numpy import array



def compute_totalerror (b,x,y,n):
    total_error = 0
    for i in range(n):
        total_error += (y[i]-(x[i,:]*b))**2
    return total_error/float(n)

def step_gradient(x,y,b,a):
    b_gradient = zeros(8)
    b_gradient = np.transpose(b_gradient)
    N= len(y)
    for i in range(N):
        for j in range(len(b_gradient)):
            b_gradient[j] += (-2/N)*x[i,j]*(y[i]-(x[i,:]*b))
    b = b -(a*b_gradient)
    return b
    

def gradient_descent_runner(x,y,b,a,num_iterations,n):
    for i in range(num_iterations):
        b = step_gradient(x,y,b,a)
        error = compute_totalerror(b,x,y,n)
        print(error)
    return b

def run ():
    df = pd.read_csv("Book1.csv")
    x= df.iloc[:,[0,1,2,3,4,5,6,7]].values
    y = df.iloc[:,[8]].values
    print(x[0,0])
    n = len(x)
    for i in range(n):
        if x[i,0] == 'M':
            x[i,0] = 1
        if x[i,0] == 'F':
            x[i,0] = 2
        if x[i,0] == 'I':
            x[i,0]= 3
    print(x[:,0])
    # hyperparameter alpha
    a = 0.01
    b = zeros(8)
    num_iterations = 1000
    b = gradient_descent_runner(x,y,b,a,num_iterations,n)
    error = compute_totalerror(b,x,y,n)
    print(b)
    print(error)
    




if __name__ == '__main__':
    run()
