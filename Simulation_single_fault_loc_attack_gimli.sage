#!/usr/bin/env sage

import os
import sys
import shutil
import time
import re
import pickle
import random as pyrandom
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import skew
from scipy.stats import norm
from scipy.stats import ttest_ind


from PRESENT_library_SIFA import *

import operator
from collections import OrderedDict

################### SCA-NFA Attack ###########################

def pause():
	programPause = input("Press the <ENTER> key to continue...")

def stop():
	sys.exit("Execution Stopped by the User...")

def bitarrtoint(X, len=4):
	X_int = 0	
	for i in range(len):
		if (X[i] == 1):
			X_int = X_int + 2^((len-1)-i)
	return X_int

def inttobitarr(X, len=4):
    l = []
    for i in range(len):
        l.append(1 if 2^i & X else 0)
    l = map(GF(2),l)
    l = list(l)
    X = list(reversed(l))
    return X

def compute_log_likelihood(x, mu, sd):
	ll = 0

	LL = np.sum(norm.logpdf(x, mu, sd))
    
	for i in x:
		ll += np.log(norm.pdf(i, mu, sd))

	ll = LL
	return ll
		
				


def three_shared_G(X1, X2, X3):
	x1 = X1[0]
	y1 = X1[1]
	z1 = X1[2]
	w1 = X1[3]
	
	x2 = X2[0]
	y2 = X2[1]
	z2 = X2[2]
	w2 = X2[3]
	
	x3 = X3[0]
	y3 = X3[1]
	z3 = X3[2]
	w3 = X3[3]
	
	one = pr._base(1)
	
	g13 = y2 + z2 + w2
	g12 = one + y2 + z2
	g11 = one + x2 + z2 + y2*w2 + y2*w3 + y3*w2 + z2*w2 + z2*w3 + z3*w2
	g10 = one + w2 + x2*y2 + x2*y3 + x3*y2 + x2*z2 + x2*z3 + x3*z2 + y2*z2 + y2*z3 + y3*z2
	
	g23 = y3 + z3 + w3
	g22 = y3 + z3
	g21 = x3 + z3 + y3*w3 + y1*w3 + y3*w1 + z3*w3 + z1*w3 + z3*w1
	g20 = w3 + x3*y3 + x1*y3 + x3*y1 + x3*z3 + x1*z3 + x3*z1 + y3*z3 + y1*z3 + y3*z1
	
	g33 = y1 + z1 + w1
	g32 = y1 + z1
	g31 = x1 + z1 + y1*w1 + y1*w2 + y2*w1 + z1*w1 + z1*w2 + z2*w1
	g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1
	
	return [g13, g12, g11, g10], [g23, g22, g21, g20], [g33, g32, g31, g30]
		
def three_shared_F(X1, X2, X3):	
	x1 = X1[0]
	y1 = X1[1]
	z1 = X1[2]
	w1 = X1[3]
	
	x2 = X2[0]
	y2 = X2[1]
	z2 = X2[2]
	w2 = X2[3]
	
	x3 = X3[0]
	y3 = X3[1]
	z3 = X3[2]
	w3 = X3[3]
	
	f13 = y2 + z2 + w2 + x2*w2 + x2*w3 + x3*w2
	f12 = x2 + z2*w2 + z2*w3 + z3*w2
	f11 = y2 + z2 + x2*w2 + x2*w3 + x3*w2
	f10 = z2 + y2*w2 + y2*w3 + y3*w2
	
	f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
	f22 = x3 + z3*w3 + z1*w3 + z3*w1
	f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
	f20 = z3 + y3*w3 + y1*w3 + y3*w1
	
	f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2*w1
	f32 = x1 + z1*w1 + z1*w2 + z2*w1
	f31 = y1 + z1 + x1*w1 + x1*w2 + x2*w1
	f30 = z1 + y1*w1 + y1*w2 + y2*w1	
 
	return [f13, f12, f11, f10], [f23, f22, f21, f20], [f33, f32, f31, f30]

def unshared_G(X):
	x = X[0]
	y = X[1]
	z = X[2]
	w = X[3]
	
	one = pr._base(1)
	
	g3 = y + z + w
	g2 = one + y + z 
	g1 = one + x + z + y*w + z*w
	g0 = one + w + x*y + x*z + y*z
	
	return [g3, g2, g1, g0]

def unshared_F(X):
	x = X[0]
	y = X[1]
	z = X[2]
	w = X[3]
	
	f3 = y + z + w + x*w
	f2 = x + z*w
	f1 = y + z + x*w
	f0 = z + y*w
	
	return [f3, f2, f1, f0]	

def three_shared_G_faulted(X1, X2, X3, fault_loc = 0):
	
	x1 = X1[0]
	y1 = X1[1]
	z1 = X1[2]
	w1 = X1[3]
	
	x2 = X2[0]
	y2 = X2[1]
	z2 = X2[2]
	w2 = X2[3]
	
	x3 = X3[0]
	y3 = X3[1]
	z3 = X3[2]
	w3 = X3[3]
	
	one = pr._base(1)
	
	
	# Compute each bit one-by-one
	
	g13 = y2 + z2 + w2
	g23 = y3 + z3 + w3
	g33 = y1 + z1 + w1
	
	g12 = one + y2 + z2
	g22 = y3 + z3
	g32 = y1 + z1
	
	g11 = one + x2 + z2 + y2*w2 + y2*w3 + y3*w2 + z2*w2 + z2*w3 + z3*w2
	g21 = x3 + z3 + y3*w3 + y1*w3 + y3*w1 + z3*w3 + z1*w3 + z3*w1
	g31 = x1 + z1 + y1*w1 + y1*w2 + y2*w1 + z1*w1 + z1*w2 + z2*w1
	
	g10 = one + w2 + x2*y2 + x2*y3 + x3*y2 + x2*z2 + x2*z3 + x3*z2 + y2*z2 + y2*z3 + y3*z2
	g20 = w3 + x3*y3 + x1*y3 + x3*y1 + x3*z3 + x1*z3 + x3*z1 + y3*z3 + y1*z3 + y3*z1
	g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1
	
	if (fault_loc == 0):
		x3_f = 1
		g10 = one + w2 + x2*y2 + x2*y3 + x3_f*y2 + x2*z2 + x2*z3 + x3_f*z2 + y2*z2 + y2*z3 + y3*z2
		g20 = w3 + x3_f*y3 + x1*y3 + x3_f*y1 + x3_f*z3 + x1*z3 + x3_f*z1 + y3*z3 + y1*z3 + y3*z1
		g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1
	elif (fault_loc == 1):
		y3_f = 1
		g10 = one + w2 + x2*y2 + x2*y3_f + x3*y2 + x2*z2 + x2*z3 + x3*z2 + y2*z2 + y2*z3 + y3_f*z2
		g20 = w3 + x3*y3_f + x1*y3_f + x3*y1 + x3*z3 + x1*z3 + x3*z1 + y3_f*z3 + y1*z3 + y3_f*z1
		g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1
	elif (fault_loc == 2):
		z3_f = 1
		g10 = one + w2 + x2*y2 + x2*y3 + x3*y2 + x2*z2 + x2*z3_f + x3*z2 + y2*z2 + y2*z3_f + y3*z2
		g20 = w3 + x3*y3 + x1*y3 + x3*y1 + x3*z3_f + x1*z3_f + x3*z1 + y3*z3_f + y1*z3_f + y3*z1
		g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1		
	elif (fault_loc == 3):
		y3_f = 1
		g11 = one + x2 + z2 + y2*w2 + y2*w3 + y3_f*w2 + z2*w2 + z2*w3 + z3*w2
		g21 = x3 + z3 + y3_f*w3 + y1*w3 + y3_f*w1 + z3*w3 + z1*w3 + z3*w1
		g31 = x1 + z1 + y1*w1 + y1*w2 + y2*w1 + z1*w1 + z1*w2 + z2*w1
	else:
		print("Error!!")			
	
	#g11 = one + x2 + z2 + y2_f*w2 + y2_f*w3 + y3*w2 + z2*w2 + z2*w3 + z3*w2
	#g31 = x1 + z1 + y1*w1 + y1*w2 + y2_f*w1 + z1*w1 + z1*w2 + z2*w1
	return [g13, g12, g11, g10], [g23, g22, g21, g20], [g33, g32, g31, g30]

def three_shared_F_faulted_new(X1, X2, X3, fault_loc = 0):	
	x1 = X1[0]
	y1 = X1[1]
	z1 = X1[2]
	w1 = X1[3]
	
	x2 = X2[0]
	y2 = X2[1]
	z2 = X2[2]
	w2 = X2[3]
	
	x3 = X3[0]
	y3 = X3[1]
	z3 = X3[2]
	w3 = X3[3]
	
	f13 = y2 + z2 + w2 + x2*w2 + x2*w3 + x3*w2
	f12 = x2 + z2*w2 + z2*w3 + z3*w2
	f11 = y2 + z2 + x2*w2 + x2*w3 + x3*w2
	f10 = z2 + y2*w2 + y2*w3 + y3*w2
	
	f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
	f22 = x3 + z3*w3 + z1*w3 + z3*w1
	f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
	f20 = z3 + y3*w3 + y1*w3 + y3*w1
	
	f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2*w1
	f32 = x1 + z1*w1 + z1*w2 + z2*w1
	f31 = y1 + z1 + x1*w1 + x1*w2 + x2*w1
	f30 = z1 + y1*w1 + y1*w2 + y2*w1
	
	"""
	if (fault_loc == 0):
		w2_f = 1
		f10 = z2 + y2*w2_f + y2*w3 + y3*w2_f
		f20 = z3 + y3*w3 + y1*w3 + y3*w1
		f30 = z1 + y1*w1 + y1*w2_f + y2*w1
	elif (fault_loc == 1):
		w2_f = 1
		f11 = y2 + z2 + x2*w2_f + x2*w3 + x3*w2_f
		f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
		f31 = y1 + z1 + x1*w1 + x1*w2_f + x2*w1
	elif (fault_loc == 2):
		w2_f = 1
		f12 = x2 + z2*w2_f + z2*w3 + z3*w2_f
		f22 = x3 + z3*w3 + z1*w3 + z3*w1
		f32 = x1 + z1*w1 + z1*w2_f + z2*w1
	elif (fault_loc == 3):
		x2_f = 1
		f13 = y2 + z2 + w2 + x2_f*w2 + x2_f*w3 + x3*w2
		f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
		f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2_f*w1
	else:
		print("Error!")		
	"""		

	if (fault_loc == 0):
		w2_f = w2 + 1
		f10 = z2 + y2*w2_f + y2*w3 + y3*w2_f
		f20 = z3 + y3*w3 + y1*w3 + y3*w1
		f30 = z1 + y1*w1 + y1*w2_f + y2*w1
		
		f11 = y2 + z2 + x2*w2_f + x2*w3 + x3*w2_f
		f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
		f31 = y1 + z1 + x1*w1 + x1*w2_f + x2*w1		

		f12 = x2 + z2*w2_f + z2*w3 + z3*w2_f
		f22 = x3 + z3*w3 + z1*w3 + z3*w1
		f32 = x1 + z1*w1 + z1*w2_f + z2*w1		
		
		f13 = y2 + z2 + w2_f + x2*w2_f + x2*w3 + x3*w2_f
		f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
		f33 = y1 + z1 + w1 + x1*w1 + x1*w2_f + x2*w1				
	elif (fault_loc == 1):
		z2_f = z2 + 1
		#f11 = y2 + z2_f + x2*w2 + x2*w3 + x3*w2
		#f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
		#f31 = y1 + z1 + x1*w1 + x1*w2 + x2*w1
		
		f10 = z2_f + y2*w2 + y2*w3 + y3*w2
		f20 = z3 + y3*w3 + y1*w3 + y3*w1
		f30 = z1 + y1*w1 + y1*w2 + y2*w1
		
		f11 = y2 + z2_f + x2*w2 + x2*w3 + x3*w2
		f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
		f31 = y1 + z1 + x1*w1 + x1*w2 + x2*w1		

		f12 = x2 + z2_f*w2 + z2_f*w3 + z3*w2
		f22 = x3 + z3*w3 + z1*w3 + z3*w1
		f32 = x1 + z1*w1 + z1*w2 + z2_f*w1		
		
		f13 = y2 + z2_f + w2 + x2*w2 + x2*w3 + x3*w2
		f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
		f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2*w1		
	elif (fault_loc == 2):
		y2_f = y2 + 1
		z2_f = z2 
		f12 = x2 + z2_f*w2 + z2_f*w3 + z3*w2
		f22 = x3 + z3*w3 + z1*w3 + z3*w1
		f32 = x1 + z1*w1 + z1*w2 + z2_f*w1
		
		f10 = z2_f + y2_f*w2 + y2_f*w3 + y3*w2
		f20 = z3 + y3*w3 + y1*w3 + y3*w1
		f30 = z1 + y1*w1 + y1*w2 + y2_f*w1
		
		f11 = y2_f + z2_f + x2*w2 + x2*w3 + x3*w2
		f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
		f31 = y1 + z1 + x1*w1 + x1*w2 + x2*w1		

		f12 = x2 + z2_f*w2 + z2_f*w3 + z3*w2
		f22 = x3 + z3*w3 + z1*w3 + z3*w1
		f32 = x1 + z1*w1 + z1*w2 + z2_f*w1		
		
		f13 = y2_f + z2_f + w2 + x2*w2 + x2*w3 + x3*w2
		f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
		f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2*w1	
	elif (fault_loc == 3):
		x2_f = x2 + 1
		z2_f = z2 
		f13 = y2 + z2_f + w2 + x2_f*w2 + x2_f*w3 + x3*w2
		f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
		f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2_f*w1
		
		f10 = z2_f + y2*w2 + y2*w3 + y3*w2
		f20 = z3 + y3*w3 + y1*w3 + y3*w1
		f30 = z1 + y1*w1 + y1*w2 + y2*w1
		
		f11 = y2 + z2_f + x2_f*w2 + x2_f*w3 + x3*w2
		f21 = y3 + z3 + x3*w3 + x1*w3 + x3*w1
		f31 = y1 + z1 + x1*w1 + x1*w2 + x2_f*w1		

		f12 = x2_f + z2_f*w2 + z2_f*w3 + z3*w2
		f22 = x3 + z3*w3 + z1*w3 + z3*w1
		f32 = x1 + z1*w1 + z1*w2 + z2_f*w1		
		
		f13 = y2 + z2_f + w2 + x2_f*w2 + x2_f*w3 + x3*w2
		f23 = y3 + z3 + w3 + x3*w3 + x1*w3 + x3*w1
		f33 = y1 + z1 + w1 + x1*w1 + x1*w2 + x2_f*w1		
	else:
		print("Error!")		
		
	return [f13, f12, f11, f10], [f23, f22, f21, f20], [f33, f32, f31, f30]

def Mej(u1,u2,u3):
	v = u1*u2 + u2*u3 + u1*u3
	if ( u1 == u2 == u3):
		flg = 0
	else:
		flg = 1
	return v, flg		

def dom_indp (x_0, x_1, y_0, y_1, z):       # first order
    t0 = x_0*y_0
    t1 = x_0*y_1
    t1 = t1 + z
    t2 = x_1*y_0
    t2 = t2 + z
    t3 = x_1*y_1
    
    q0 = t0 + t1
    q1 = t2 + t3
    
    return q0, q1

def sni_refresh(x_0, x_1, r):      # First order
    #r = pr._base(pyrandom.randint(0,1))
    y_0 = x_0 + r
    y_1 = x_1 + r 
    return y_0, y_1

def present_dom_sbox(X0, X1, Z, fault_loc = None):
    
    # Unmasked Equations
    #Y[0] = (X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0] + X[1]*X[2]*X[3] + X[1]*X[2] + X[2] + X[3] + 1) 
    #Y[1] = (X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0]*X[2] + X[0]*X[3] + X[0] + X[1] + X[2]*X[3] + 1) 
    #Y[2] = (X[0]*X[1]*X[3] + X[0]*X[1] + X[0]*X[2]*X[3] + X[0]*X[2] + X[0] + X[1]*X[2]*X[3] + X[2]) 
    #Y[3] = (X[0] + X[1] + X[3] + X[1]*X[2])

    x00 = X0[0]
    x01 = X0[1]
    x02 = X0[2]
    x03 = X0[3]
    
    x10 = X1[0]
    x11 = X1[1]
    x12 = X1[2]
    x13 = X1[3]    
    
    #-----------------------------------
    # Inject Fault 
    #-----------------------------------
    if fault_loc is not None:
        if (fault_loc == 0):
            x00 = x00 + 1
        elif (fault_loc == 1):
            x01 = x01 + 1
        elif (fault_loc == 2):
            x02 = x02 + 1
        elif (fault_loc == 3):
            x03 = x03 + 1
        else:
            print("error")
            stop()    
    else:
         pass                   
    #-----------------------------------
    
    # for debugging...
    #X = [0]*4
    
    #for i in range(4):
    #    X[i] = X0[i] + X1[i]
    
    #In0 = X[0]*X[1]*X[3]
    #In1 = X[0]*X[2]*X[3]
    #In2 = X[1]*X[2]*X[3]
    #In3 = X[1]*X[2]
    #In4 = X[0]*X[2]
    #In5 = X[0]*X[3]
    #In6 = X[2]*X[3]
    #In7 = X[0]*X[1]
    #----
    
    y00 = None
    y01 = None
    y02 = None
    y03 = None
    
    y10 = None
    y11 = None
    y12 = None
    y13 = None
    
    #z0 = pr._base(pyrandom.randint(0,1))     #1
    z0 = Z[0]
    T0, T1 = dom_indp(x00, x10, x01, x11, z0) # X[0]*X[1]
    rn = Z[1]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          # SNI Refresh #2  
    #z1 = pr._base(pyrandom.randint(0,1))     #3
    z1 = Z[2]
    T2, T3 = dom_indp(T0_r, T1_r, x03, x13, z1)   # X[0]*X[1]*X[3]
    rn = Z[3]
    T2_r, T3_r = sni_refresh(T2, T3, rn)          # SNI Refresh #4  
    
    
    y00 = T2_r
    y10 = T3_r
       
    rn = Z[4]
    T2_r, T3_r = sni_refresh(T2_r, T3_r, rn)          # SNI Refresh #5 
    
    y01 = T2_r
    y11 = T3_r
          
    rn = Z[5]
    T2_r, T3_r = sni_refresh(T2_r, T3_r, rn)          # SNI Refresh #6
    
    y02 = T2_r
    y12 = T3_r
     

    

    #z0 = pr._base(pyrandom.randint(0,1))      #7
    z0 = Z[6]
    T0, T1 = dom_indp(x00, x10, x02, x12, z0) # X[0]*X[2]
    rn = Z[7]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          # SNI Refresh #8  
    #z1 = pr._base(pyrandom.randint(0,1))      #9
    z1 = Z[8] 
    T2, T3 = dom_indp(T0_r, T1_r, x03, x13, z1)   # X[0]*X[2]*X[3]
    rn = Z[9]
    T2_r, T3_r = sni_refresh(T2, T3, rn)          # X[0]*X[2]*X[3] #10   



    y00 = y00 + T2_r
    y10 = y10 + T3_r
      
    rn = Z[10]
    T2_r, T3_r = sni_refresh(T2_r, T3_r, rn)          # SNI Refresh #11
    
    y01 = y01 + T2_r
    y11 = y11 + T3_r
    
    rn = Z[11]
    T2_r, T3_r = sni_refresh(T2_r, T3_r, rn)          # SNI Refresh #12
    
    y02 = y02 + T2_r
    y12 = y12 + T3_r
    
    

    #z0 = pr._base(pyrandom.randint(0,1))        #13
    z0 = Z[12]
    T0, T1 = dom_indp(x01, x11, x02, x12, z0) # X[1]*X[2]
    rn = Z[13]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          #SNI Refresh #14  
    #z1 = pr._base(pyrandom.randint(0,1))        #15
    z1 = Z[14]
    T2, T3 = dom_indp(T0_r, T1_r, x03, x13, z1)   # X[1]*X[2]*X[3]
    rn = Z[15]
    T2_r, T3_r = sni_refresh(T2, T3, rn)          # X[1]*X[2]*X[3]  #16   
      
    
    y00 = y00 + T2_r
    y10 = y10 + T3_r
    
    rn = Z[16]
    T2_r, T3_r = sni_refresh(T2_r, T3_r, rn)          # SNI Refresh #17
    
    y02 = y02 + T2_r
    y12 = y12 + T3_r


    #z0 = pr._base(pyrandom.randint(0,1))  #18
    z0 = Z[17]
    T0, T1 = dom_indp(x01, x11, x02, x12, z0) # X[1]*X[2]
    rn = Z[18]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          #SNI Refresh #19     
    
    
    y00 = y00 + T0_r
    y10 = y10 + T1_r
    
    rn = Z[19]
    T0_r, T1_r = sni_refresh(T0_r, T1_r, rn)          #SNI Refresh #20 
    
    y03 = T0_r
    y13 = T1_r

    
    
    #z0 = pr._base(pyrandom.randint(0,1))    #21
    z0 = Z[20]
    T0, T1 = dom_indp(x00, x10, x02, x12, z0) # X[0]*X[2]
    rn = Z[21]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          #SNI Refresh #22  
           
    
    y01 = y01 + T0_r
    y11 = y11 + T1_r
         
    rn = Z[22]
    T0_r, T1_r = sni_refresh(T0_r, T1_r, rn)          # SNI Refresh #23 
    
    y02 = y02 + T0_r
    y12 = y12 + T1_r
             
        
    #z0 = pr._base(pyrandom.randint(0,1))   # 24
    z0 = Z[23]
    T0, T1 = dom_indp(x00, x10, x03, x13, z0) # X[0]*X[3]
    rn = Z[24]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          #SNI Refresh #25          
      
    
    y01 = y01 + T0_r
    y11 = y11 + T1_r
    
    
    
    #z0 = pr._base(pyrandom.randint(0,1))     #26
    z0 = Z[25]
    T0, T1 = dom_indp(x02, x12, x03, x13, z0) # X[2]*X[3]
    rn = Z[26]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          #SNI Refresh #27    
    
    
    y01 = y01 + T0_r
    y11 = y11 + T1_r
    
    
    #z0 = pr._base(pyrandom.randint(0,1))  #28
    z0 = Z[27]
    T0, T1 = dom_indp(x00, x10, x01, x11, z0) # X[0]*X[1]
    rn = Z[28]
    T0_r, T1_r = sni_refresh(T0, T1, rn)          # SNI Refresh #29  


    y02 = y02 + T0_r
    y12 = y12 + T1_r            
    
       
    
    y00 = y00 + x00 + x02 + x03 + 1           # X[0] + X[2] + X[3] + 1 
    y10 = y10 + x10 + x12 + x13
    
    y01 = y01 + x00 + x01 + 1                 # X[0] + X[1] + 1
    y11 = y11 + x10 + x11
    
    y02 = y02 + x00 + x02                     # X[0] + X[2]
    y12 = y12 + x10 + x12

    y03 = y03 + x00 + x01 + x03               # X[0] + X[1] + X[3]
    y13 = y13 + x10 + x11 + x13   
    
    
    
    Y0 = [y00, y01, y02, y03]
    Y1 = [y10, y11, y12, y13]
      
    return Y0, Y1


# Daemen's SIFA Protected S-Box
def Clone (v, v_):
	v_ = v
	return v_

def lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = None):
	#print("here")

	r0 = None
	r1 = None 
	s0 = None
	s1 = None 
	t0 = None 
	t1 = None
	
	
	Rr_ = None
	Rr_ = Clone(Rr, Rr_)
	
	Rt_ = None
	Rt_ = Clone(Rt, Rt_)
	
	Rs = Rr_ + Rt_		#
	
	if (fault_loc == 0):
		c0 = c0 + 1 		# fault
	elif (fault_loc == 1):
		b0 = b0 + 1			# fault
	elif (fault_loc == 2):
		a1 = a1 + 1			# fault

	
	b0_ = None
	b0_ = Clone (b0, b0_)

	c1_ = None
	c1_ = Clone (c1, c1_)	
		
	T0 = (b0_+ 1) * c1_		#
	
	
	a1_ = None
	a1_ = Clone (a1, a1_)	

	b1_ = None
	b1_ = Clone (b1, b1_)		
	 
	T2 = a1_ * b1_		# 
	 
	
	b0_ = None
	b0_ = Clone (b0, b0_)	
	 
	c0_ = None
	c0_ = Clone (c0, c0_)
	
	T1 = (b0_ + 1) * c0_		#
	
	
	a1_ = None
	a1_ = Clone (a1, a1_)	
	 
	b0_ = None
	b0_ = Clone (b0, b0_)
	
	T3 = a1_ * b0_		#	
	
	
	
	Rr_ = None
	Rr_ = Clone(Rr, Rr_)	
	
	r0 = T0 + Rr_		#
	

	Rt_ = None
	Rt_ = Clone(Rt, Rt_)	
	
	t1 = T2 + Rt_		#	


	r0 = r0 + T1		#
	
	t1 = t1 + T3		#
	
	
	
	
	c0_ = None
	c0_ = Clone (c0, c0_)
	
	a1_ = None
	a1_ = Clone (a1, a1_)	
	
	T0 = (c0_ + 1)*a1_	#


	b1_ = None
	b1_ = Clone (b1, b1_)
	
	c1_ = None
	c1_ = Clone (c1, c1_)	
	
	T2 = b1_ * c1_		#
	

	c0_ = None
	c0_ = Clone (c0, c0_)
	
	a0_ = None
	a0_ = Clone (a0, a0_)	
	
	T1 = (c0_ + 1)*a0_	#
	

	b1_ = None
	b1_ = Clone (b1, b1_)
	
	c0_ = None
	c0_ = Clone (c0, c0_)	
	
	T3 = b1_ * c0_		#



	Rs_ = None
	Rs_ = Clone(Rs, Rs_)
	
	s0 = T0 + Rs_		#
	
	r1 = T2 + Rr		#
	
	s0 = s0 + T1		#
	
	r1 = r1 + T3		#
	


	a0_ = None
	a0_ = Clone (a0, a0_)
	
	b1_ = None
	b1_ = Clone (b1, b1_)
	
	T0 = (a0_ + 1)*b1_	#	
	
	c1_ = None
	c1_ = Clone (c1, c1_)
	
	a1_ = None
	a1_ = Clone (a1, a1_)
	
	T2 = c1_ * a1_		#
	

	a0_ = None
	a0_ = Clone (a0, a0_)
	
	b0_ = None
	b0_ = Clone (b0, b0_)
	
	T1 = (a0_ + 1)*b0_	#		
	 

	c1_ = None
	c1_ = Clone (c1, c1_)
	
	a0_ = None
	a0_ = Clone (a0, a0_)
	
	T3 = c1_ * a0_		#
	
	
	

	t0 = T0 + Rt		#
	
	s1 = T2 + Rs		#
	
	t0 = t0 + T1		#
	
	s1 = s1 + T3		#
	
	r0 = r0 + a0		#
	
	t1 = t1 + c1		#
	
	s0 = s0 + b0		#
	
	r1 = r1 + a1		#
	
	t0 = t0 + c0		#
	
	s1 = s1 + b1		#

	r = r0 + r1
	s = s0 + s1
	t = t0 + t1
	
	a = a0 + a1
	b = b0 + b1
	c = c0 + c1
	
	#print((a, b, c), (r, s, t))
	#print(bitarrtoint([a, b, c], len=3), bitarrtoint([r, s, t], len=3))

	return r0, r1, s0, s1, t0, t1





       

# Gimli
def rot(x, n):
    """Bitwise rotation (to the left) of n bits considering the \
    string of bits is 32 bits long"""
    x %= 1 << 32
    n %= 32
    # if n == 0:
    # print(hex(x), "=>", hex((x >> (32 - n)) | (x << n) % (1 << 32)))
    return (x >> (32 - n)) | (x << n) % (1 << 32)

def dom_indp_word(X_0, X_1, Y_0, Y_1, Z):
    T0 = X_0 & Y_0
    T1 = X_0 & Y_1
    T1 = T1 ^^ Z
    T2 = X_1 & Y_0
    T2 = T2 ^^ Z
    T3 = X_1 & Y_1
    
    q0 = T0 ^^ T1
    q1 = T2 ^^ T3 
    
    return q0, q1

def gimli_perm_unprotected(state_ip, bit_cont=False, fault_loc=None, fround=None, fault_column=None):
    x = 0
    y = 0
    z = 0
    state = [0]*12
    for i in range(12):
        state[i] = state_ip[i]
    maskval = int("0xffffffff", 16)
    rconst = int("0x9e377900", 16)
    
    tracked_bit = None
    
    for round in reversed(range(1, 25)):
        for column in range(4):
            x = rot(state[    column], 24)
            y = rot(state[4 + column],  9)
            z =        state[8 + column]
            
            if fault_loc is not None:
                fmask = int("0x01000000", 16)
                if (round == fround):
                    #print(round)
                    if (fault_loc == 0):
                        z = z ^^ fmask
                    elif (fault_loc == 1):
                        y = y ^^ fmask 
                    elif (fault_loc == 2):
                        x = x ^^ fmask
                    else:
                        pass
                    #    print("error!!")
                    #    stop()
                    #pause()    
                else:
                    #print(round)
                    pass             
            else:
                pass    
            
            if ( (fault_loc == 4) and (round == fround) and (column == fault_column) ): 
                fmask = int("0x00010000", 16)
                #fmask = int("0x00000000", 16)
                #print "{0:08x}".format(y)
                #print(inttobitarr(y, len=32))
                #print "{0:08x}".format(z)
                #print(inttobitarr(z, len=32))
                state[8 + column] = ( x ^^ (z << 1) ^^ (( y & (z^^fmask)) << 2) ) & maskval
                #print "{0:08x}".format(state[8 + column]) 
                #print(inttobitarr(state[8 + column], len=32))
                #pause()                                
            else:        
                state[8 + column] = ( x ^^ (z << 1) ^^ ((y&z) << 2) ) & maskval
                
            if ( (fault_loc == 3) and (round == fround) and (column == fault_column) ):
                fmask = int("0x00000040", 16)
                #fmask = int("0x00000000", 16)
                #state[4 + column] = ( y ^^ x        ^^ ( (x^^ fmask|(z))  << 1) ) & maskval 
                state[4 + column] = ( y ^^ x        ^^ ( (x|(z))  << 1) ) & maskval & int("0xFFFFFF7F", 16) 
                #print "{0:08x}".format(state[4 + column]) 
                #print(inttobitarr(state[4 + column], len=32))
                #pause()
            else:
                state[4 + column] = ( y ^^ x        ^^ ((x|z) << 1) ) & maskval     
            state[column]     = (z ^^ y        ^^ ((x&y) << 3) ) & maskval 

        #if (round == fround):
        #    print("at fault round")
        #    print("----------------------")
        #    for l in range(12):
        #        print "{0:08x}".format(state[l]), 
        #        if (i % 4 == 3):
        #            print("")
        #    print("----------------------")

        if ((round & 3) == 0):          # small swap: pattern s...s...s... etc.
            x = state[0]
            state[0] = state[1]
            state[1] = x
            x = state[2]
            state[2] = state[3]
            state[3] = x
        
        if ((round & 3) == 2):       # big swap: pattern ..S...S...S. etc.
            x = state[0]
            state[0] = state[2]
            state[2] = x
            x = state[1]
            state[1] = state[3]
            state[3] = x
        

        if ((round & 3) == 0):      # add constant: pattern c...c...c... etc.
            state[0] = (state[0] ^^ (rconst | round)) & maskval

        if (round == fround):
            #print("at fault round")
            #print("----------------------")
            #for l in range(12):
            #    print "{0:08x}".format(state[l]), 
            #    if (i % 4 == 3):
            #        print("")
            #print("----------------------")
            #print("")
            #print "{0:08x}".format(state[8])
            #print(inttobitarr(state[8], len=32))
            if bit_cont is False:
                pass
            else:
               tracked_bit = inttobitarr(state[4], len=32)[31 - 7]    
            #print("")
            #print(tracked_bit)
            #pause()            

    return state, tracked_bit

def ANF_OR(x, y):
    return (x + y + x*y)

def binary_SEI(x, n):
    x = float(x)
    n = float(n)
    #print(x)
    #print(n)
    sei = ( ((x/n) - 0.5)^2 + (((n-x)/n) - 0.5)^2 )
    if (sei < 0):
        sei = -1*sei
    #print(sei)
    #pause()
    return sei

def cal_HW(n):
    c = 0
    while n:
        c += 1
        n &= n - 1

    return c

def ECC_word(a, b, c):
    
    noise_mean = 33.0
    #noise_std = 6.20
    noise_std = 3.0
    
    t1 = (a&b)
    t2 = (b&c)
    t3 = (c&a) 
    d = t1 ^^ t2 ^^ t3
    
    # Calulate leakage
    #if (a != b):
    #    print(hex(a), hex(b))
    #    print(hex(a ^^ b))
        #pause()
    
    #leak = cal_HW(t1 & int("0xffffffff", 16)) + np.random.normal(noise_mean, noise_std, 1)
    leak = cal_HW(t1 & int("0xffffffff", 16)) 
    #leak = (a^^b)

    #print(leak)
    #pause()
    
    t1 = (a&b)
    t2 = (b&c)
    t3 = (c&a) 
    e = t1 ^^ t2 ^^ t3
    
    t1 = (a&b)
    t2 = (b&c)
    t3 = (c&a) 
    f = t1 ^^ t2 ^^ t3    
    
    return d, e, f, leak

def gimli_perm_DOM(state0, state1, bit_cont=False, leak_measure=False, fault_loc=None, fround=None, fault_column=None):
    
    leakage_measurement = None
    x00 = 0
    y00 = 0
    z00 = 0
    
    x10 = 0
    y10 = 0
    z10 = 0
    
    state_00 = [0]*12
    state_10 = [0]*12
    
    for i in range(12):
        state_00[i] = state0[i]
        state_10[i] = state1[i]
    
    test_state = [0]*12
    for i in range(12):
        test_state[i] = state_00[i] ^^ state_10[i]
        
    #for i in range(12):
    #    print("{0:08x}".format(test_state[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #print("")
    #pause()    
    
    x01 = 0
    y01 = 0
    z01 = 0
    
    x11 = 0
    y11 = 0
    z11 = 0
    
    state_01 = [0]*12
    state_11 = [0]*12
    
    for i in range(12):
        state_01[i] = state0[i]
        state_11[i] = state1[i]

    x02 = 0
    y02 = 0
    z02 = 0
    
    x12 = 0
    y12 = 0
    z12 = 0
    
    state_02 = [0]*12
    state_12 = [0]*12
    
    for i in range(12):
        state_02[i] = state0[i]
        state_12[i] = state1[i]    
    
        
    maskval = int("0xffffffff", 16)
    rconst = int("0x9e377900", 16)
    
    tracked_bit = None    
    debug_state = -1
    for round in reversed(range(1, 25)):
        #if (round == 22):
        #    print(round)
        #    for i in range(12):
        #        print("{0:08x}".format(state_00[i] ^^ state_10[i])),
        #        if (i % 4 == 3):
        #            print("")
        #    print("----------------------")
        #    print("")
            #pause()
        
        for column in range(4):
            x00 = rot(state_00[    column], 24)
            #if ( (fault_loc == 4) and (round == fround) and (column == fault_column) ):
            #    x00 = pyrandom.randint(0,4294967295)
                    
            y00 = rot(state_00[4 + column],  9)
            z00 =        state_00[8 + column]
            
            x10 = rot(state_10[    column], 24)
            y10 = rot(state_10[4 + column],  9)
            z10 =        state_10[8 + column]            
            
            x01 = rot(state_01[    column], 24)
            y01 = rot(state_01[4 + column],  9)
            z01 =        state_01[8 + column]
            
            x11 = rot(state_11[    column], 24)
            y11 = rot(state_11[4 + column],  9)
            z11 =        state_11[8 + column]         
            
            x02 = rot(state_02[    column], 24)
            y02 = rot(state_02[4 + column],  9)
            z02 =        state_02[8 + column]
            
            x12 = rot(state_12[    column], 24)
            y12 = rot(state_12[4 + column],  9)
            z12 =        state_12[8 + column]                                 
            
            if fault_loc is not None:
                fmask = int("0x01000000", 16)
                if (round == fround):
                    #print(round)
                    if (fault_loc == 0):
                        z00 = z00 ^^ fmask
                    elif (fault_loc == 1):
                        y00 = y00 ^^ fmask 
                    elif (fault_loc == 2):
                        x00 = x00 ^^ fmask
                    else:
                        pass
                    #    print("error!!")
                    #    stop()
                    #pause()    
                else:
                    #print(round)
                    pass             
            else:
                pass    
            
            if ( (fault_loc == 4) and (round == fround) and (column == fault_column) ): 
                #print "{0:08x}".format(y)
                #print(inttobitarr(y, len=32))
                #print "{0:08x}".format(z)
                #print(inttobitarr(z, len=32))
                r = pyrandom.randint(0,4294967295)
                #fmask = int("0x00000000", 16)
                #fmask = int("0x00010000", 16)
                #z00_f = z00 ^^ fmask       # single-bit flip
                #z00_f = int("0x00000000", 16)   # instruction-skip (32-bit) and local
                #print(z00)
                #pause()
                #x00 = int("0x00000000", 16)      # instruction-skip (32-bit) and global
                
                #tst = pyrandom.randint(0,1)
                #if (tst == 0):
                x00 = pyrandom.randint(0,4294967295)      # instruction-skip (32-bit) and global
                #x00 = 0
                #else:
                #    pass    
                
                #print(x00)
                #print(z00_f ^^ z00)
                #q00, q10 = dom_indp_word(z00_f, z10, y00, y10, r)  # instruction-skip (32-bit) and local
                q00, q10 = dom_indp_word(z00, z10, y00, y10, r)         
                state_00[8 + column] = ( x00 ^^ (z00 << 1) ^^ (q00 << 2) ) & maskval
                state_10[8 + column] = ( x10 ^^ (z10 << 1) ^^ (q10 << 2) ) & maskval
                #print(state_00[8 + column])
                #print("here")
                #pause()        
                #state[8 + column] = ( x ^^ (z << 1) ^^ (( y & (z^^fmask)) << 2) ) & maskval
                #print "{0:08x}".format(state[8 + column]) 
                #print(inttobitarr(state[8 + column], len=32))

                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(z01, z11, y01, y11, r)
                state_01[8 + column] = ( x01 ^^ (z01 << 1) ^^ (q01 << 2) ) & maskval
                state_11[8 + column] = ( x11 ^^ (z11 << 1) ^^ (q11 << 2) ) & maskval
                #print(state_01[8 + column])
                
                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(z02, z12, y02, y12, r)
                state_02[8 + column] = ( x02 ^^ (z02 << 1) ^^ (q02 << 2) ) & maskval
                state_12[8 + column] = ( x12 ^^ (z12 << 1) ^^ (q12 << 2) ) & maskval                

                #if (state_00[8 + column] != state_01[8 + column]):
                #    print(hex(q00), hex(q10))
                #    print(hex(q01), hex(q11))
                #    print(state_00[8 + column], state_01[8 + column])
                #    pause()

            else:
                r = pyrandom.randint(0,4294967295)
                q00, q10 = dom_indp_word(z00, z10, y00, y10, r)
                state_00[8 + column] = ( x00 ^^ (z00 << 1) ^^ (q00 << 2) ) & maskval
                state_10[8 + column] = ( x10 ^^ (z10 << 1) ^^ (q10 << 2) ) & maskval
                #pause()        
                #state[8 + column] = ( x ^^ (z << 1) ^^ ((y&z) << 2) ) & maskval
                
                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(z01, z11, y01, y11, r)
                state_01[8 + column] = ( x01 ^^ (z01 << 1) ^^ (q01 << 2) ) & maskval
                state_11[8 + column] = ( x11 ^^ (z11 << 1) ^^ (q11 << 2) ) & maskval
                
                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(z02, z12, y02, y12, r)
                state_02[8 + column] = ( x02 ^^ (z02 << 1) ^^ (q02 << 2) ) & maskval
                state_12[8 + column] = ( x12 ^^ (z12 << 1) ^^ (q12 << 2) ) & maskval                
                
                
            if ( (fault_loc == 3) and (round == fround) and (column == fault_column) ):
                #fmask = int("0x00000040", 16)
                #fmask = int("0x00000000", 16)
                r = pyrandom.randint(0,4294967295)
                q00, q10 = dom_indp_word(z00, z10, x00, x10, r)
                q00 = q00 ^^ z00 ^^ x00
                q10 = q10 ^^ z10 ^^ x10
                state_00[4 + column] = ( y00 ^^ x00        ^^ ( q00  << 1) ) & maskval & int("0xFFFFFF7F", 16)
                state_10[4 + column] = ( y10 ^^ x10        ^^ ( q10  << 1) ) & maskval 
                #state[4 + column] = ( y ^^ x        ^^ ( (x^^ fmask|(z))  << 1) ) & maskval 
                #state[4 + column] = ( y ^^ x        ^^ ( (x|(z))  << 1) ) & maskval & int("0xFFFFFF7F", 16) 
                #print "{0:08x}".format(state[4 + column]) 
                #print(inttobitarr(state[4 + column], len=32))
                #pause()
                
                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(z01, z11, x01, x11, r)
                q01 = q01 ^^ z01 ^^ x01
                q11 = q11 ^^ z11 ^^ x11
                state_01[4 + column] = ( y01 ^^ x01        ^^ ( q01  << 1) ) & maskval
                state_11[4 + column] = ( y11 ^^ x11        ^^ ( q11  << 1) ) & maskval 

                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(z02, z12, x02, x12, r)
                q02 = q02 ^^ z02 ^^ x02
                q12 = q12 ^^ z12 ^^ x12
                state_02[4 + column] = ( y02 ^^ x02        ^^ ( q02  << 1) ) & maskval
                state_12[4 + column] = ( y12 ^^ x12        ^^ ( q12  << 1) ) & maskval
            elif ( (fault_loc == 4) and (round == fround) and (column == fault_column) ):
                #print("here")
                #print(x00)
                r = pyrandom.randint(0,4294967295)
                #z00 = int("0x00000000", 16)         # instruction-skip (32-bit global)
                q00, q10 = dom_indp_word(z00, z10, x00, x10, r)
                q00 = q00 ^^ z00 ^^ x00
                q10 = q10 ^^ z10 ^^ x10
                state_00[4 + column] = ( y00 ^^ x00        ^^ ( q00  << 1) ) & maskval
                state_10[4 + column] = ( y10 ^^ x10        ^^ ( q10  << 1) ) & maskval                 
                #state[4 + column] = ( y ^^ x        ^^ ((x|z) << 1) ) & maskval     
                #pause()
                
                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(z01, z11, x01, x11, r)
                q01 = q01 ^^ z01 ^^ x01
                q11 = q11 ^^ z11 ^^ x11
                state_01[4 + column] = ( y01 ^^ x01        ^^ ( q01  << 1) ) & maskval
                state_11[4 + column] = ( y11 ^^ x11        ^^ ( q11  << 1) ) & maskval 

                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(z02, z12, x02, x12, r)
                q02 = q02 ^^ z02 ^^ x02
                q12 = q12 ^^ z12 ^^ x12
                state_02[4 + column] = ( y02 ^^ x02        ^^ ( q02  << 1) ) & maskval
                state_12[4 + column] = ( y12 ^^ x12        ^^ ( q12  << 1) ) & maskval                                                               
            else:
                r = pyrandom.randint(0,4294967295)
                q00, q10 = dom_indp_word(z00, z10, x00, x10, r)
                q00 = q00 ^^ z00 ^^ x00
                q10 = q10 ^^ z10 ^^ x10
                state_00[4 + column] = ( y00 ^^ x00        ^^ ( q00  << 1) ) & maskval
                state_10[4 + column] = ( y10 ^^ x10        ^^ ( q10  << 1) ) & maskval                 
                #state[4 + column] = ( y ^^ x        ^^ ((x|z) << 1) ) & maskval     
                #pause()
                
                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(z01, z11, x01, x11, r)
                q01 = q01 ^^ z01 ^^ x01
                q11 = q11 ^^ z11 ^^ x11
                state_01[4 + column] = ( y01 ^^ x01        ^^ ( q01  << 1) ) & maskval
                state_11[4 + column] = ( y11 ^^ x11        ^^ ( q11  << 1) ) & maskval 

                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(z02, z12, x02, x12, r)
                q02 = q02 ^^ z02 ^^ x02
                q12 = q12 ^^ z12 ^^ x12
                state_02[4 + column] = ( y02 ^^ x02        ^^ ( q02  << 1) ) & maskval
                state_12[4 + column] = ( y12 ^^ x12        ^^ ( q12  << 1) ) & maskval                      
            
            if ( (fault_loc == 4) and (round == fround) and (column == fault_column) ):
                #print(x00)
                r = pyrandom.randint(0,4294967295)
                #z00 = int("0x00000000", 16)             # instruction-skip (32-bit global)
                q00, q10 = dom_indp_word(x00, x10, y00, y10, r)
                state_00[column]     = (z00 ^^ y00        ^^ (q00 << 3) ) & maskval
                state_10[column]     = (z10 ^^ y10        ^^ (q10 << 3) ) & maskval     
                #state[column]     = (z ^^ y        ^^ ((x&y) << 3) ) & maskval 
                
                #print(column)
                #tmp_checker = state_00[column] ^^ state_10[column]
                #print "{0:08x}".format(tmp_checker) 
                #pause()
                #print(state_00[column])
                #pause()
                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(x01, x11, y01, y11, r)
                state_01[column]     = (z01 ^^ y01        ^^ (q01 << 3) ) & maskval
                state_11[column]     = (z11 ^^ y11        ^^ (q11 << 3) ) & maskval 
                
                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(x02, x12, y02, y12, r)
                state_02[column]     = (z02 ^^ y02        ^^ (q02 << 3) ) & maskval
                state_12[column]     = (z12 ^^ y12        ^^ (q12 << 3) ) & maskval                       
            else:    
                r = pyrandom.randint(0,4294967295)
                q00, q10 = dom_indp_word(x00, x10, y00, y10, r)
                state_00[column]     = (z00 ^^ y00        ^^ (q00 << 3) ) & maskval
                state_10[column]     = (z10 ^^ y10        ^^ (q10 << 3) ) & maskval     
                #state[column]     = (z ^^ y        ^^ ((x&y) << 3) ) & maskval 
                
                #tmp_checker = state_00[column] ^^ state_10[column]
                #print "{0:08x}".format(tmp_checker) 
                #pause()
                
                #r = pyrandom.randint(0,2^32)
                q01, q11 = dom_indp_word(x01, x11, y01, y11, r)
                state_01[column]     = (z01 ^^ y01        ^^ (q01 << 3) ) & maskval
                state_11[column]     = (z11 ^^ y11        ^^ (q11 << 3) ) & maskval 
                
                #r = pyrandom.randint(0,2^32)
                q02, q12 = dom_indp_word(x02, x12, y02, y12, r)
                state_02[column]     = (z02 ^^ y02        ^^ (q02 << 3) ) & maskval
                state_12[column]     = (z12 ^^ y12        ^^ (q12 << 3) ) & maskval                            

        #if (round == fround):
            #print("here")
            #print("at fault round")
            #print("----------------------")
            #for l in range(12):
            #    dum = state_00[l]^^state_10[l]
            #    print "{0:08x}".format(dum), 
            #    if (l % 4 == 3):
            #        print("")
            #print("----------------------")
            

        if ((round & 3) == 0):          # small swap: pattern s...s...s... etc.
            x00 = state_00[0]
            state_00[0] = state_00[1]
            state_00[1] = x00
            x00 = state_00[2]
            state_00[2] = state_00[3]
            state_00[3] = x00
            
            x01 = state_01[0]
            state_01[0] = state_01[1]
            state_01[1] = x01
            x01 = state_01[2]
            state_01[2] = state_01[3]
            state_01[3] = x01
            
            x02 = state_02[0]
            state_02[0] = state_02[1]
            state_02[1] = x02
            x02 = state_02[2]
            state_02[2] = state_02[3]
            state_02[3] = x02            

        if ((round & 3) == 0):          # small swap: pattern s...s...s... etc.
            x10 = state_10[0]
            state_10[0] = state_10[1]
            state_10[1] = x10
            x10 = state_10[2]
            state_10[2] = state_10[3]
            state_10[3] = x10
            
            x11 = state_11[0]
            state_11[0] = state_11[1]
            state_11[1] = x11
            x11 = state_11[2]
            state_11[2] = state_11[3]
            state_11[3] = x11
            
            x12 = state_12[0]
            state_12[0] = state_12[1]
            state_12[1] = x12
            x12 = state_12[2]
            state_12[2] = state_12[3]
            state_12[3] = x12                        
        
        if ((round & 3) == 2):       # big swap: pattern ..S...S...S. etc.
            x00 = state_00[0]
            state_00[0] = state_00[2]
            state_00[2] = x00
            x00 = state_00[1]
            state_00[1] = state_00[3]
            state_00[3] = x00
            
            x01 = state_01[0]
            state_01[0] = state_01[2]
            state_01[2] = x01
            x01 = state_01[1]
            state_01[1] = state_01[3]
            state_01[3] = x01
            
            x02 = state_02[0]
            state_02[0] = state_02[2]
            state_02[2] = x02
            x02 = state_02[1]
            state_02[1] = state_02[3]
            state_02[3] = x02                        
            
        if ((round & 3) == 2):       # big swap: pattern ..S...S...S. etc.
            x10 = state_10[0]
            state_10[0] = state_10[2]
            state_10[2] = x10
            x10 = state_10[1]
            state_10[1] = state_10[3]
            state_10[3] = x10  
            
            x11 = state_11[0]
            state_11[0] = state_11[2]
            state_11[2] = x11
            x11 = state_11[1]
            state_11[1] = state_11[3]
            state_11[3] = x11
            
            x12 = state_12[0]
            state_12[0] = state_12[2]
            state_12[2] = x12
            x12 = state_12[1]
            state_12[1] = state_12[3]
            state_12[3] = x12                                  
        
        if ((round & 3) == 0):      # add constant: pattern c...c...c... etc.
            state_00[0] = (state_00[0] ^^ (rconst | round)) & maskval  
            state_01[0] = (state_01[0] ^^ (rconst | round)) & maskval     
            state_02[0] = (state_02[0] ^^ (rconst | round)) & maskval               

        # Error Correction at the end of round on shares....
        for v in range(12):
           #if (state_00[v] != state_01[v]):
           #    print(hex(state_00[v]), hex(state_01[v])) 
           #    pause()
           d0, d1, d2, leak0 = ECC_word(state_00[v], state_01[v], state_02[v]) 
           state_00[v] = d0
           state_01[v] = d1
           state_02[v] = d2
           
           if ((round == fround) and (leak_measure is True) and (v == 2)):
               leakage_measurement = leak0
               #if (leak0 != 0):
               #    print(leak0)
               #    print(v)
               #    pause()
           
           d0, d1, d2, leak1 = ECC_word(state_10[v], state_11[v], state_12[v]) 
           state_10[v] = d0
           state_11[v] = d1
           state_12[v] = d2                 
        #pause()
        #if ((round == fround) and leak_measure is True):
        #    leakage_measurement = leak0
        #    if (leak0 != 0):
        #        print(leak0)
        #        pause()
            #print("here")
                
        #if (round == fround):
            #print("----------------------")
            #state_dummy0 = [0]*12
            #state_dummy1 = [0]*12
            #state_dummy2 = [0]*12
            #for d in range(12):
            #    state_dummy0[d] = state_00[d] ^^ state_10[d]
            #    state_dummy1[d] = state_01[d] ^^ state_11[d]
            #    state_dummy2[d] = state_02[d] ^^ state_12[d]
            #for l in range(12):
            #    print "{0:08x}".format(state_dummy0[l]), "{0:08x}".format(state_dummy1[l]), "{0:08x}".format(state_dummy2[l])
            #    print "{0:08x}".format(state_dummy0[l] ^^ state_dummy1[l])
            #    if (l % 4 == 3):
            #        print("")
            #print("----------------------")
                  
            
            
            #for v in range(12):
            #   d0, d1, d2 = ECC_word(state_00[v], state_01[v], state_02[v]) 
            #   state_00[v] = d0
            #   state_01[v] = d1
            #   state_02[v] = d2
               
            #   d0, d1, d2 = ECC_word(state_10[v], state_11[v], state_12[v]) 
            #   state_10[v] = d0
            #   state_11[v] = d1
            #   state_12[v] = d2               
           
            
            #print("")
            #print("")
            #print("After correction...")
            #print("----------------------")
            #print("----------------------")
            
            #state_dummy0 = [0]*12
            #state_dummy1 = [0]*12
            #state_dummy2 = [0]*12
            #for d in range(12):
            #    state_dummy0[d] = state_00[d] ^^ state_10[d]
            #    state_dummy1[d] = state_01[d] ^^ state_11[d]
            #    state_dummy2[d] = state_02[d] ^^ state_12[d]
            #for l in range(12):
            #    print "{0:08x}".format(state_dummy0[l]), "{0:08x}".format(state_dummy1[l]), "{0:08x}".format(state_dummy2[l])
            #    if (l % 4 == 3):
            #        print("")
            #print("----------------------")        
            #pause()
        #-----------------------------------------

        if (round == fround):
            #print("at fault round")
            #print("----------------------")
            #state_dummy = [0]*12
            #for d in range(12):
            #    state_dummy[d] = state_00[d] ^^ state_10[d]
            #for l in range(12):
            #    print "{0:08x}".format(state_dummy[l]), 
            #    if (l % 4 == 3):
            #        print("")
            #print("----------------------")
            #print("")
            #print "{0:08x}".format(state_dummy[8])
            #print(inttobitarr(state_dummy[8], len=32))
            if bit_cont is False:
                pass
            else:
               tracked_bit = inttobitarr(state_dummy[4], len=32)[31 - 7]    
            #print("")
            #print(tracked_bit)
            #pause()          
    
    return state_00, state_10, leakage_measurement



pr = PRESENT(80,31) 
all_val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
sbox_tab = pr.sbox()








#--------------------------------------------
# Non-profiled attack (correction must)
#--------------------------------------------
"""
fault_sig_prob = 1.0
fault_loc = 0
num_enc = 8000
key = 7
key_bits = inttobitarr(key,len=3)
#print(key_bits)

exp_repeat = 10

HW_array = []
CT_array = []

#-----------------------------------------------------------------------
# 3-bit S-Box followed by a key addition
# New testing for single fault and correcton
#-----------------------------------------------------------------------
for ne in range(num_enc):
    
    x = pyrandom.randint(0,7)
    
    m1 = pyrandom.randint(0,7)
    rr = pyrandom.randint(0,1)
    rt = pyrandom.randint(0,1)
        
    M1 = inttobitarr(m1, len=3)
    X = inttobitarr(x, len=3)
    X1 = [X[i] + M1[i] for i in range(3)]
    X2 = M1
    Rr = inttobitarr(rr, len=1)
    Rt = inttobitarr(rt, len=1)

             
    a0 = X1[0]
    a1 = X2[0]

    b0 = X1[1]
    b1 = X2[1]

    c0 = X1[2]
    c1 = X2[2]

    Rr = Rr[0]
    Rt = Rt[0]


    # Make some noise (fault noise)
    rand_fault_loc = pyrandom.randint(0,3) # choose a fault location randomly
    rn = sage.misc.prandom.random()
    if (rn < fault_sig_prob):		#signal
        r0, r1, s0, s1, t0, t1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = fault_loc)
    else:							#noise
        r0, r1, s0, s1, t0, t1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = rand_fault_loc)	
    #r0, r1, s0, s1, t0, t1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = fault_loc)
    r0_c, r1_c, s0_c, s1_c, t0_c, t1_c = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt)

    r0_c1, r1_c1, s0_c1, s1_c1, t0_c1, t1_c1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt)


    ####### Error Correction ############
    ar0 = r0*r0_c 
    br0 = r0_c*r0_c1
    cr0 = r0*r0_c1

    r0_ec = ar0 + br0 + cr0

    ar1 = r1*r1_c 
    br1 = r1_c*r1_c1
    cr1 = r1*r1_c1

    r1_ec = ar1 + br1 + cr1


    as0 = s0*s0_c 
    bs0 = s0_c*s0_c1
    cs0 = s0*s0_c1

    s0_ec = as0 + bs0 + cs0

    as1 = s1*s1_c 
    bs1 = s1_c*s1_c1
    cs1 = s1*s1_c1

    s1_ec = as1 + bs1 + cs1


    at0 = t0*t0_c 
    bt0 = t0_c*t0_c1
    ct0 = t0*t0_c1

    t0_ec = at0 + bt0 + ct0

    at1 = t1*t1_c 
    bt1 = t1_c*t1_c1
    ct1 = t1*t1_c1

    t1_ec = at1 + bt1 + ct1	

    DS_inter = [at1, bt1, ct1, at0, bt0, ct0, as1, bs1, cs1, as0, bs0, cs0, ar1, br1, cr1, ar0, br0, cr0]

    # Leakage Measurement
    ######################################
    hw = 0
    for b in DS_inter:
        if (b == 1):
            hw = hw + 1
            
    
    # Add noise in HWs--------
    #print(hw)
    #hw = hw + np.random.normal(noise_mean, noise_std, 1)
    #print(hw)
    #----- HW dependent noise -----#
    #hw = hw + np.random.normal(noise_mean_per_hw[hw], noise_std_per_hw[hw], 1)
    #pause()
    #-------------------------		    
    HW_array.append(hw)

    corr_op = [t1_ec, t0_ec, s1_ec, s0_ec, r1_ec, r0_ec]

    for_check = [t1_c1, t0_c1, s1_c1, s0_c1, r1_c1, r0_c1]

    # key addition
    corr_op[0] = corr_op[0] + key_bits[0]
    corr_op[2] = corr_op[2] + key_bits[1]
    corr_op[4] = corr_op[4] + key_bits[2]



    corr_op_unmasked = [(corr_op[0]+corr_op[1]), (corr_op[2]+corr_op[3]), (corr_op[4]+corr_op[5])]
    #for_check_unmasked = [(for_check[0]+for_check[1]), (for_check[2]+for_check[3]), (for_check[4]+for_check[5])]

    #print(corr_op)
    #print(for_check)

    #print(corr_op_unmasked)
    #print(for_check_unmasked)


    #print(x)
    #print(fault_loc)
    #print(DS_inter)
    #pause()

    ######################################

    ciphertext = bitarrtoint(corr_op_unmasked,len=3)
    CT_array.append(ciphertext)
    


# Attack Phase
#           0  1, 2, 3, 4, 5, 6, 7
sbox_tab = [0, 5, 6, 2, 3, 4, 1, 7]
inv_sbox_tab = [0, 6, 3, 4, 5, 1, 2, 7]



key_HW_dict = {}
for kg in range(8):
    HW_hypo = []
    for ct in CT_array:
        intm_1 = ct^^kg 
        intm_2 = inv_sbox_tab[intm_1]
        
        x = intm_2
        
        m1 = pyrandom.randint(0,7)
        rr = pyrandom.randint(0,1)
        rt = pyrandom.randint(0,1)
            
        M1 = inttobitarr(m1, len=3)
        X = inttobitarr(x, len=3)
        X1 = [X[i] + M1[i] for i in range(3)]
        X2 = M1
        Rr = inttobitarr(rr, len=1)
        Rt = inttobitarr(rt, len=1)

                 
        a0 = X1[0]
        a1 = X2[0]

        b0 = X1[1]
        b1 = X2[1]

        c0 = X1[2]
        c1 = X2[2]

        Rr = Rr[0]
        Rt = Rt[0]


        # Make some noise (fault noise)
        rand_fault_loc = pyrandom.randint(0,3) # choose a fault location randomly
        rn = sage.misc.prandom.random()
        if (rn < fault_sig_prob):		#signal
            r0, r1, s0, s1, t0, t1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = fault_loc)
        else:							#noise
            r0, r1, s0, s1, t0, t1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = rand_fault_loc)	
        #r0, r1, s0, s1, t0, t1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt, fault_loc = fault_loc)
        r0_c, r1_c, s0_c, s1_c, t0_c, t1_c = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt)

        r0_c1, r1_c1, s0_c1, s1_c1, t0_c1, t1_c1 = lam_S_box (a0, a1, b0, b1, c0, c1, Rr, Rt)


        ####### Error Correction ############
        ar0 = r0*r0_c 
        br0 = r0_c*r0_c1
        cr0 = r0*r0_c1

        r0_ec = ar0 + br0 + cr0

        ar1 = r1*r1_c 
        br1 = r1_c*r1_c1
        cr1 = r1*r1_c1

        r1_ec = ar1 + br1 + cr1


        as0 = s0*s0_c 
        bs0 = s0_c*s0_c1
        cs0 = s0*s0_c1

        s0_ec = as0 + bs0 + cs0

        as1 = s1*s1_c 
        bs1 = s1_c*s1_c1
        cs1 = s1*s1_c1

        s1_ec = as1 + bs1 + cs1


        at0 = t0*t0_c 
        bt0 = t0_c*t0_c1
        ct0 = t0*t0_c1

        t0_ec = at0 + bt0 + ct0

        at1 = t1*t1_c 
        bt1 = t1_c*t1_c1
        ct1 = t1*t1_c1

        t1_ec = at1 + bt1 + ct1	

        DS_inter = [at1, bt1, ct1, at0, bt0, ct0, as1, bs1, cs1, as0, bs0, cs0, ar1, br1, cr1, ar0, br0, cr0]

        # Leakage Measurement
        ######################################
        hw = 0
        for b in DS_inter:
            if (b == 1):
                hw = hw + 1
                
        HW_hypo.append(hw)

    key_HW_dict[kg]  = tuple(HW_hypo)

#print(key_HW_dict)

# Correlation

corr_arr = {}
for key in range(8):
    t1 = list(key_HW_dict[key])
    len_t1 = len(t1)
    
    t2 = list(HW_array)
    len_t2 = len(t2)
    
    if (len_t1 > len_t2):
        T1 = t1[:len_t2]
        T2 = t2
    else:
        T1 = t1
        T2 = t2[:len_t1] 
    sim_data = np.array(T1)
    trace_data = np.array(T2)              
    corr_arr[key] = np.corrcoef(sim_data, trace_data)[0,1]
max_corr = -999
max_key = -1
for k in corr_arr.keys():
    if (corr_arr[k] > max_corr):
        max_corr = corr_arr[k]
        max_key = k
print(max_key)
print(max_corr)

sorted_tuples = sorted(corr_arr.items(), key=operator.itemgetter(1), reverse=True)
sorted_dict = OrderedDict()
for k, v in sorted_tuples:
    sorted_dict[k] = v

print(sorted_dict.keys())
"""


#---------------------------------------------------
# Non-profiled attack (PRESENT DOM; correction must)
#---------------------------------------------------

# Sbox and inv SBox tables
"""
present_sbox = SBox([12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2], big_endian=True)

inv_sbox = [0 for _ in range(2^4)]
for i in range(2^4):
    inv_sbox[present_sbox[i]] = i
inv_sbox = SBox(inv_sbox)
"""

# Test DOM AND
"""
x = pr._base(0)
y = pr._base(0)

q = x*y

print(x, y)
print(q)
x_0 = pyrandom.randint(0,1)
x_0 = pr._base(x_0)
x_1 = x + x_0

y_0 = pyrandom.randint(0,1)
y_0 = pr._base(y_0)
y_1 = y + y_0

z = pr._base(pyrandom.randint(0,1))

print(x_0, x_1)
print(y_0, y_1)

q_0, q_1 = dom_indp (x_0, x_1, y_0, y_1, z)
print(q_0, q_1)
q = q_0 + q_1
print(q)
pause()
"""


# Test DOM PRESENT SBox (We have put a lot of refresh gadgets to ensure composition; maybe it can be optimized)

"""
for x in range(16):
    m1 = pyrandom.randint(0,15)

        
    M1 = inttobitarr(m1, len=4)
    X = inttobitarr(x, len=4)
    X0 = [X[i] + M1[i] for i in range(4)] # share 0
    X1 = M1                               # share 1


    #print(X)
    #print(M1)
    #print(X1)
    #print(X2)

    Y = [0]*4

    Y[0] = (X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0] + X[1]*X[2]*X[3] + X[1]*X[2] + X[2] + X[3] + 1) 
    Y[1] = (X[0]*X[1]*X[3] + X[0]*X[2]*X[3] + X[0]*X[2] + X[0]*X[3] + X[0] + X[1] + X[2]*X[3] + 1) 
    Y[2] = (X[0]*X[1]*X[3] + X[0]*X[1] + X[0]*X[2]*X[3] + X[0]*X[2] + X[0] + X[1]*X[2]*X[3] + X[2]) 
    Y[3] = (X[0] + X[1] + X[3] + X[1]*X[2]) 

    y = bitarrtoint(Y, len=4)
    
    Y0 = [0]*4
    Y1 = [0]*4
    
    Yu = [0]*4
    
    
    Z = [pyrandom.randint(0,1) for i in range(29)] # The extra randomness required for masked computation
    Y0, Y1 = present_dom_sbox(X0, X1, Z, fault_loc = 1)

    for s in range(4):
        Yu[s] = Y0[s] + Y1[s]
    
    yu = bitarrtoint(Yu, len=4)
    
    diff = [0]*4
    for d in range(4):
       diff[d] = Y[d] + Yu[d]
        
    print(y)
    print(yu)
    print(diff)
    print("")
    pause()
"""


"""
fault_sig_prob = 1.0
fault_loc = 0
num_enc = 10000
key = 5
key_bits = inttobitarr(key,len=4)
#print(key_bits)


exp_repeat = 10

HW_array = []
CT_array = []

for ne in range(num_enc):
    
    x = pyrandom.randint(0,15)
    m1 = pyrandom.randint(0,15)

        
    M1 = inttobitarr(m1, len=4)
    X = inttobitarr(x, len=4)
    X0 = [X[i] + M1[i] for i in range(4)] # share 0
    X1 = M1                               # share 1    
    
    Y0 = [0]*4
    Y1 = [0]*4
    
    Z = [pyrandom.randint(0,1) for i in range(29)] # The extra randomness required for masked computation
    
    # Make some noise (fault noise)
    rand_fault_loc = pyrandom.randint(0,3) # choose a fault location randomly #3 is not a fault loc, but leads to correct execution which is as noise.
    rn = sage.misc.prandom.random()
    if (rn < fault_sig_prob):		#signal
        Y0, Y1 = present_dom_sbox(X0, X1, Z, fault_loc = fault_loc)
    else:							#noise
        Y0, Y1 = present_dom_sbox(X0, X1, Z, fault_loc = rand_fault_loc)
    Y0_c, Y1_c = present_dom_sbox(X0, X1, Z)
    Y0_c1, Y1_c1 = present_dom_sbox(X0, X1, Z)

    ####### Error Correction ############
    
    y00, y01, y02, y03 = Y0
    y10, y11, y12, y13 = Y1
    
    y00_c, y01_c, y02_c, y03_c = Y0_c
    y10_c, y11_c, y12_c, y13_c = Y1_c

    y00_c1, y01_c1, y02_c1, y03_c1 = Y0_c1
    y10_c1, y11_c1, y12_c1, y13_c1 = Y1_c1    
    
    
    ay00 = y00*y00_c 
    by00 = y00_c*y00_c1
    cy00 = y00*y00_c1
    
    y00_ec = ay00 + by00 + cy00
    
    ay10 = y10*y10_c 
    by10 = y10_c*y10_c1
    cy10 = y10*y10_c1
    
    y10_ec = ay10 + by10 + cy10
    

    ay01 = y01*y01_c 
    by01 = y01_c*y01_c1
    cy01 = y01*y01_c1
    
    y01_ec = ay01 + by01 + cy01
    
    ay11 = y11*y11_c 
    by11 = y11_c*y11_c1
    cy11 = y11*y11_c1
    
    y11_ec = ay11 + by11 + cy11
                

    ay02 = y02*y02_c 
    by02 = y02_c*y02_c1
    cy02 = y02*y02_c1
    
    y02_ec = ay02 + by02 + cy02
    
    ay12 = y12*y12_c 
    by12 = y12_c*y12_c1
    cy12 = y12*y12_c1
    
    y12_ec = ay12 + by12 + cy12

    ay03 = y03*y03_c 
    by03 = y03_c*y03_c1
    cy03 = y03*y03_c1
    
    y03_ec = ay03 + by03 + cy03
    
    ay13 = y13*y13_c 
    by13 = y13_c*y13_c1
    cy13 = y13*y13_c1
    
    y13_ec = ay13 + by13 + cy13


    DS_inter = [ay00, by00, cy00, ay01, by01, cy01, ay02, by02, cy02, ay03, by03, cy03, ay10, by10, cy10, ay11, by11, cy11, ay12, by12, cy12, ay13, by13, cy13]

    # Leakage Measurement
    ######################################
    hw = 0
    for b in DS_inter:
        if (b == 1):
            hw = hw + 1
            
    
    # Add noise in HWs--------
    #print(hw)
    #hw = hw + np.random.normal(noise_mean, noise_std, 1)
    #print(hw)
    #----- HW dependent noise -----#
    #hw = hw + np.random.normal(noise_mean_per_hw[hw], noise_std_per_hw[hw], 1)
    #pause()
    #-------------------------		    
    HW_array.append(hw)
    
    corr_op = [y00_ec, y10_ec, y01_ec, y11_ec, y02_ec, y12_ec, y03_ec, y13_ec]
    
    
    # key addition
    corr_op[0] = corr_op[0] + key_bits[0]
    corr_op[2] = corr_op[2] + key_bits[1]
    corr_op[4] = corr_op[4] + key_bits[2]
    corr_op[6] = corr_op[6] + key_bits[3]

    corr_op_unmasked = [(corr_op[0]+corr_op[1]), (corr_op[2]+corr_op[3]), (corr_op[4]+corr_op[5]), (corr_op[6]+corr_op[7])]

    ciphertext = bitarrtoint(corr_op_unmasked,len=4)
    
    #print(ciphertext)
    #print(present_sbox(x)^^key)
    #pause()
    CT_array.append(ciphertext)


# Attack Phase
key_HW_dict = {}
for kg in range(16):
    HW_hypo = []
    for ct in CT_array:
        intm_1 = ct^^kg 
        intm_2 = inv_sbox[intm_1]
        
        x = intm_2
        m1 = pyrandom.randint(0,15)

            
        M1 = inttobitarr(m1, len=4)
        X = inttobitarr(x, len=4)
        X0 = [X[i] + M1[i] for i in range(4)] # share 0
        X1 = M1                               # share 1    
        
        Y0 = [0]*4
        Y1 = [0]*4
        
        Z = [pyrandom.randint(0,1) for i in range(29)] # The extra randomness required for masked computation
        
        # Make some noise (fault noise)
        rand_fault_loc = pyrandom.randint(0,3) # choose a fault location randomly 
        rn = sage.misc.prandom.random()
        if (rn < fault_sig_prob):       #signal
            Y0, Y1 = present_dom_sbox(X0, X1, Z, fault_loc = fault_loc)
        else:                           #noise
            Y0, Y1 = present_dom_sbox(X0, X1, Z, fault_loc = rand_fault_loc)
        Y0_c, Y1_c = present_dom_sbox(X0, X1, Z)
        Y0_c1, Y1_c1 = present_dom_sbox(X0, X1, Z)

        ####### Error Correction ############
        
        y00, y01, y02, y03 = Y0
        y10, y11, y12, y13 = Y1
        
        y00_c, y01_c, y02_c, y03_c = Y0_c
        y10_c, y11_c, y12_c, y13_c = Y1_c

        y00_c1, y01_c1, y02_c1, y03_c1 = Y0_c1
        y10_c1, y11_c1, y12_c1, y13_c1 = Y1_c1    
        
        
        ay00 = y00*y00_c 
        by00 = y00_c*y00_c1
        cy00 = y00*y00_c1
        
        y00_ec = ay00 + by00 + cy00
        
        ay10 = y10*y10_c 
        by10 = y10_c*y10_c1
        cy10 = y10*y10_c1
        
        y10_ec = ay10 + by10 + cy10
        

        ay01 = y01*y01_c 
        by01 = y01_c*y01_c1
        cy01 = y01*y01_c1
        
        y01_ec = ay01 + by01 + cy01
        
        ay11 = y11*y11_c 
        by11 = y11_c*y11_c1
        cy11 = y11*y11_c1
        
        y11_ec = ay11 + by11 + cy11
                    

        ay02 = y02*y02_c 
        by02 = y02_c*y02_c1
        cy02 = y02*y02_c1
        
        y02_ec = ay02 + by02 + cy02
        
        ay12 = y12*y12_c 
        by12 = y12_c*y12_c1
        cy12 = y12*y12_c1
        
        y12_ec = ay12 + by12 + cy12

        ay03 = y03*y03_c 
        by03 = y03_c*y03_c1
        cy03 = y03*y03_c1
        
        y03_ec = ay03 + by03 + cy03
        
        ay13 = y13*y13_c 
        by13 = y13_c*y13_c1
        cy13 = y13*y13_c1
        
        y13_ec = ay13 + by13 + cy13


        DS_inter = [ay00, by00, cy00, ay01, by01, cy01, ay02, by02, cy02, ay03, by03, cy03, ay10, by10, cy10, ay11, by11, cy11, ay12, by12, cy12, ay13, by13, cy13]

        # Leakage Measurement
        ######################################
        hw = 0
        for b in DS_inter:
            if (b == 1):
                hw = hw + 1
                
        
        # Add noise in HWs--------
        #print(hw)
        #hw = hw + np.random.normal(noise_mean, noise_std, 1)
        #print(hw)
        #----- HW dependent noise -----#
        #hw = hw + np.random.normal(noise_mean_per_hw[hw], noise_std_per_hw[hw], 1)
        #pause()
        #-------------------------
        HW_hypo.append(hw)
    
    key_HW_dict[kg]  = tuple(HW_hypo)
    
# Correlation

corr_arr = {}
for key in range(16):
    t1 = list(key_HW_dict[key])
    len_t1 = len(t1)
    
    t2 = list(HW_array)
    len_t2 = len(t2)
    
    if (len_t1 > len_t2):
        T1 = t1[:len_t2]
        T2 = t2
    else:
        T1 = t1
        T2 = t2[:len_t1] 
    sim_data = np.array(T1)
    trace_data = np.array(T2)              
    corr_arr[key] = np.corrcoef(sim_data, trace_data)[0,1]
max_corr = -9999
max_key = -1
for k in corr_arr.keys():
    if (corr_arr[k] > max_corr):
        max_corr = corr_arr[k]
        max_key = k
print(max_key)
print(max_corr)

sorted_tuples = sorted(corr_arr.items(), key=operator.itemgetter(1), reverse=True)
sorted_dict = OrderedDict()
for k, v in sorted_tuples:
    sorted_dict[k] = v

print(sorted_dict)
print(sorted_dict.keys())
"""


# Test DOM for 32-bit words
"""
Z = pyrandom.randint(0,2^32)
X = pyrandom.randint(0,2^32)
Y = pyrandom.randint(0,2^32)

X0 = pyrandom.randint(0,2^32)
Y0 = pyrandom.randint(0,2^32)

X1 = X0 ^^ X
Y1 = Y0 ^^ Y

q0, q1 = dom_indp_word (X0, X1, Y0, Y1, Z)

O = X & Y
O_s = q0 ^^ q1

print(O)
print(O_s)
    
stop()
"""


big_num = int("0x9e3779b9", 16)
maskval = int("0xffffffff", 16)
num_sim =  5000                     # Change this parameter to increase/decrease fault simulation count
#for i in range(12):
#    state[i] = (i * i * i + i * big_num) & maskval   
#print(state)
Key_reg = [0]*8
Nonce_tmp = [0]*4
ineffective_nonce_list = []
tracked_bit_list = []
tracked_bit_calc_list = []

tracked_bit = None
tracked_bit_c = None

leakage_trace = []
leakage_nonce_dict = {}

#with open("opfile_merged.txt") as f:
#    lines_op = f.readlines()
    
with open("full_input.txt") as f:
    lines_ip = f.readlines() 
    
#with open("opfile.txt") as f:
#    lines_op = f.readlines()    

def str_to_32_bit_int(inp, length=32):      # length in #characters
    #print(inp)    
    bls = [inp[i:i+8] for i in range(0, length, 8)]
    #print(bls)
    opt = [ int(bls[i], 16) for i in range(len(bls))]
    #print(opt)
    return opt


"""
Nonce = []
Key_reg = []

op_lst = []
leakage_sim = []
error_checker = []

for t in range(num_sim):
    ip_line = lines_ip[t]
    op_line = lines_op[t]
    ip_line = ip_line.rstrip('\n')
    op_line = op_line.rstrip('\n')

    nonce = str_to_32_bit_int(ip_line[:32],length=32)
    key = str_to_32_bit_int(ip_line[32:],length=64)
    Nonce.append(nonce)
    Key_reg = key    

    leaksim = op_line[96:112]
    #print(leaksim)
    leakage_sim.append(leaksim)

print("here")


#print(op_lst)
#print(leakage_sim)

leakage_sim_a = []
leakage_sim_b = []
for leak in leakage_sim:
   leakage_sim_a.append(int(leak[:8], 16))
   leakage_sim_b.append(int(leak[8:], 16)) 

#print(leakage_sim_a)
#print(leakage_sim_b)

HW_leakage_sim_a = []
HW_leakage_sim_b = []



for l in range(num_sim):
    HW_leakage_sim_a.append( cal_HW(leakage_sim_a[l]) ) 
    HW_leakage_sim_b.append( cal_HW(leakage_sim_b[l]) ) 

#print(HW_leakage_sim_a)
#print(HW_leakage_sim_b)

leakage_nonce_dict = {}

for l in range(num_sim):
    leakage_nonce_dict[tuple(Nonce[l])] = HW_leakage_sim_b[l]
"""



for num in range(num_sim):
    
    
    # The nonce
    Nonce = [0]*4
    state = [0]*12
    
    state0 = [0]*12    #share0
    state1 = [0]*12    #share1 
    
    
    ip_line = lines_ip[num]
    ip_line = ip_line.rstrip('\n')
    
    Nonce = str_to_32_bit_int(ip_line[:32],length=32)
    Key_reg = str_to_32_bit_int(ip_line[32:],length=64)
    
    for i in range(4):
        state[i] = Nonce[i]
    
    for i in range(4, 12):
        state[i] =  Key_reg[i-4]  
 
    #for i in range(12):
    #    print("{0:08x}".format(state[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #print("")
    
    
    
    """
    for i in range(4):
        #state[i] = pyrandom.randint(0,4294967295)
        state[i] = (i + 2*big_num + 1)& maskval 
        #state[i] = 2000
        Nonce[i] = state[i]
        #print("{0:08x}".format(tmp))
        #pause()
    Nonce_tmp = [Nonce[x] for x in range(4)]    
    # The key    
    for i in range(4, 12):
        state[i] = (i + 7*big_num) & maskval 
        #state[i] = (i + 5*big_num) & maskval 
        Key_reg[i-4] = (i + 7*big_num) & maskval 
    """
    #test_state = [0x00000000, 0xba79379e, 0x7af36e3c, 0x466da6da, 0x24e7dd78, 0x1a611517, 0x2edb4cb5, 0x66558453, 0xc8cfbbf1, 0x5a4af38f, 0x22c52a2e, 0x264062cc]
    #test_state = [0x00000000, 0xba79379e, 0x7af36e3c, 0x466da6da, 0x24e7dd78, 0x1a611517, 0x2edb4cb5, 0x66558453, 0xc8cfbbf1, 0x5a4af38f, 0x22c52a2e, 0x264062cc]
    #for i in range(12):
    #    state[i] = test_state[i]

    #for i in range(12):
    #    print("{0:08x}".format(state[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #print("")
    
    ###### Process the state format to make it compatible with the code in STM2
    #for i in range(12):
    #    print("{0:08x}".format(state[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #print("")
    
    
    state_formatted = []
    for g in state:
        data = "{0:08x}".format(g)
        bs = [data[i:i+2] for i in range(0, len(data), 2)]
        bs = reversed(bs)
        result = "".join(bs)
        state_formatted.append(int(result, 16))
    
    for i in range(12):
        state[i] = state_formatted[i]   
    
    Key_reg_formatted = []
    for kr in Key_reg:
        data = "{0:08x}".format(kr)
        bs = [data[i:i+2] for i in range(0, len(data), 2)]
        bs = reversed(bs)
        result = "".join(bs)
        Key_reg_formatted.append(int(result, 16))        
 
    for i in range(8):
        Key_reg[i] = Key_reg_formatted[i]    
        
    Nonce_formatted = []
    for nr in Nonce:
        data = "{0:08x}".format(nr)
        bs = [data[i:i+2] for i in range(0, len(data), 2)]
        bs = reversed(bs)
        result = "".join(bs)
        Nonce_formatted.append(int(result, 16))        
 
    for i in range(4):
        Nonce[i] = Nonce_formatted[i]      
        
          
    #for i in range(12):
    #    print("{0:08x}".format(state[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #print("")   
    #pause()      
    ######
       
 
    
    state_copy = [0]*12
    for j in range(12):
        state_copy[j] = state[j]

    #for i in range(12):
    #    print("{0:08x}".format(state_copy[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    
    # share the states
    for i in range(12):
        state0[i] = pyrandom.randint(0,4294967295)
        state1[i] = state0[i] ^^ state[i]
    
    
    
    fround = 22
    #state, tracked_bit = gimli_perm_unprotected(state, bit_cont=True, fault_loc=3, fround=fround,fault_column=0) # successful
    #state, tracked_bit = gimli_perm_unprotected(state, bit_cont=True, fault_loc=4, fround=fround,fault_column=0)
    state_c, tracked_bit_c = gimli_perm_unprotected(state_copy, bit_cont=True)

    #debug_state = [0]*12
    state_0, state_1, leakage_measurement = gimli_perm_DOM(state0, state1, leak_measure=True, fault_loc=4, fround=fround,fault_column=0)
    #state_0, state_1 = gimli_perm_DOM(state0, state1)
    #print(leakage_measurement)
    #print("{0:08x}".format(debug_state))
    #pause()
    
    leakage_trace.append(leakage_measurement)
    leakage_nonce_dict[tuple(Nonce)] = leakage_measurement
    
    print(num)
    state_op = [0]*12

    for i in range(12):
        state_op[i] = state_0[i] ^^ state_1[i]

    cf_flag = True
    for i in range(12):
        if (state_op[i] != state_c[i]):
            cf_flag = False
        else:
            pass    
    #print(cf_flag)
    
    if (cf_flag == True):
        ineffective_nonce_list.append(Nonce)
    tracked_bit_list.append(tracked_bit)

    
    if(fround == 24):
        k0, k1, k2, k3, k4, k5, k6, k7 = Key_reg
        #print(k0, k1, k2, k3, k4, k5, k6, k7)

        n0, n1, n2, n3 = Nonce
        maskval = int("0xffffffff", 16)
        c = (int('0x9e377900', 16) | 24) & maskval


        k0_30 = inttobitarr(k0, len=32)[31 - 30]
        n0_15 = inttobitarr(n0, len=32)[31 - 15]
        n0_14 = inttobitarr(n0, len=32)[31 - 14]
        k4_6 = inttobitarr(k4, len=32)[31 - 6]

        #print(k0_30, n0_15, n0_14, k4_6)
        b_23_0_7 = k0_30 + n0_15 + ANF_OR(n0_14, k4_6)
        print(b_23_0_7)
        tracked_bit_calc_list.append(b_23_0_7)
    
    if(fround == 23):
        k0, k1, k2, k3, k4, k5, k6, k7 = Key_reg
        #print(k0, k1, k2, k3, k4, k5, k6, k7)

        n0, n1, n2, n3 = Nonce
        maskval = int("0xffffffff", 16)
        c = (int('0x9e377900', 16) | 24) & maskval
     
        k0_21 = inttobitarr(k0, len=32)[31 - 21]
        k4_29 = inttobitarr(k4, len=32)[31 - 29]
        k5_15 = inttobitarr(k5, len=32)[31 - 15]
        k1_6 = inttobitarr(k1, len=32)[31 - 6]
        k1_3 = inttobitarr(k1, len=32)[31 - 3]
        k5_14 = inttobitarr(k5, len=32)[31 - 14]
        k1_5 = inttobitarr(k1, len=32)[31 - 5]
        k1_2 = inttobitarr(k1, len=32)[31 - 2]
        k4_5 = inttobitarr(k4, len=32)[31 - 5]
        k4_4 = inttobitarr(k4, len=32)[31 - 4]
        k0_27 = inttobitarr(k0, len=32)[31 - 27]


        n0_6 = inttobitarr(n0, len=32)[31 - 6]
        #n0_15 = inttobitarr(n0, len=32)[31 - 15]
        n0_5 = inttobitarr(n0, len=32)[31 - 5]
        n1_20 = inttobitarr(n1, len=32)[31 - 20]
        n1_19 = inttobitarr(n1, len=32)[31 - 19]
        n0_14 = inttobitarr(n0, len=32)[31 - 14]

        c15 = inttobitarr(c, len=32)[31 - 15]
        c14 = inttobitarr(c, len=32)[31 - 14]


        tmp0 = k0_21 + n0_6 
        #tmp0 = k0_21 + n0_15 
        tmp1 = ANF_OR(n0_5, k4_29)
        tmp2 = k5_15 + k1_6
        tmp3 = n1_20 * k1_3
        tmp4 = c15
        tmp5 = k5_14 + k1_5
        tmp6 = n1_19 * k1_2
        tmp7 = c14
        tmp8 = n0_14 + k4_5
        tmp9 = k4_4 * k0_27

        b_22_0_7 = tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + ANF_OR( (tmp5 + tmp6 + tmp7), (tmp8 + tmp9) )   
        #print(b_22_0_7)
        tracked_bit_calc_list.append(b_22_0_7)   
    #print("")
    #pause()

    #for i in range(12):
    #    print("{0:08x}".format(state_op[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #pause()
    #for i in range(12):
    #    print("{0:08x}".format(state_c[i])),
    #    if (i % 4 == 3):
    #        print("")
    #print("----------------------")
    #pause()
#print(len(ineffective_nonce_list))
#print(leakage_trace)
#print(leakage_nonce_dict)

#print(tracked_bit_list)
#print(tracked_bit_calc_list)
#check_mismatch = True
#for i in range(len(tracked_bit_list)):
#    if (tracked_bit_list[i] != tracked_bit_calc_list[i]):
#        check_mismatch = False

#print(check_mismatch)

#stop()





# The new combined attack code (using difference of means)
key_hyp = [i for i in range(2^6)]
maskval = int("0xffffffff", 16)
c = (int('0x9e377900', 16) | 24) & maskval
#k0, k1, k2, k3, k4, k5, k6, k7 = Key_reg
#n0, n1, n2, n3 = Nonce


max_dm = -9999
max_dm_ind = -1
dm_dict = {}
for kh in key_hyp:
    ks1 = inttobitarr(kh, len=6)[0]
    ks2 = inttobitarr(kh, len=6)[1]
    ks3 = inttobitarr(kh, len=6)[2]
    k4_29 = inttobitarr(kh, len=6)[3]
    k1_3 = inttobitarr(kh, len=6)[4]
    k1_2 = inttobitarr(kh, len=6)[5]
    
    Bin0 = []
    Bin1 = []
    for non in leakage_nonce_dict.keys():
        leak = leakage_nonce_dict[non]
        non = list(non)
        n0, n1, n2, n3 = non
        n0_6 = inttobitarr(n0, len=32)[31 - 6]
        n0_5 = inttobitarr(n0, len=32)[31 - 5]
        n1_20 = inttobitarr(n1, len=32)[31 - 20]
        n1_19 = inttobitarr(n1, len=32)[31 - 19]
        n0_14 = inttobitarr(n0, len=32)[31 - 14]

        c15 = inttobitarr(c, len=32)[31 - 15]
        c14 = inttobitarr(c, len=32)[31 - 14] 
        
        tmp0 = ks1 + n0_6 
        tmp1 = ANF_OR(n0_5, k4_29)
        tmp2 = (n1_20 * k1_3) + c15
 
        tmp3 = ks2 + (n1_19 * k1_2) + c14
        tmp4 = n0_14 + ks3
        

        b_22_0_7 = tmp0 + tmp1 + tmp2 + ANF_OR( tmp3, tmp4 )
        if b_22_0_7 == 1:
            b = 1
        else:
            b = 0
        if (b == 0):
            Bin0.append(leak)
        else:
            Bin1.append(leak)                           
    
    #print(Bin0)
    #print(Bin1)
    #t, p = ttest_ind(Bin0, Bin1)
    #print(t)
    
    B0 = np.array(Bin0)
    B1 = np.array(Bin1)
    
    m0 = np.mean(B0)
    m1 = np.mean(B1)
    
    dm = (m0 - m1)
    if (dm < 0):
        dm = -1*dm
    
    if (max_dm < dm):
        max_dm = dm
        max_dm_ind = kh
    
    #print(dm)    
    dm_dict[kh] = dm    
    #pause() 

print(max_dm_ind)

sorted_tuples = sorted(dm_dict.items(), key=operator.itemgetter(1), reverse=True)
sorted_dict = {}
sorted_dict = OrderedDict()
for k, v in sorted_tuples:
    sorted_dict[k] = v

#print(sorted_dict)
print(sorted_dict.keys())

print("Actual Key:")
k0, k1, k2, k3, k4, k5, k6, k7 = Key_reg

k0_21 = inttobitarr(k0, len=32)[31 - 21]
k4_29 = inttobitarr(k4, len=32)[31 - 29]
k5_15 = inttobitarr(k5, len=32)[31 - 15]
k1_6 = inttobitarr(k1, len=32)[31 - 6]
k1_3 = inttobitarr(k1, len=32)[31 - 3]
k5_14 = inttobitarr(k5, len=32)[31 - 14]
k1_5 = inttobitarr(k1, len=32)[31 - 5]
k1_2 = inttobitarr(k1, len=32)[31 - 2]
k4_5 = inttobitarr(k4, len=32)[31 - 5]
k4_4 = inttobitarr(k4, len=32)[31 - 4]
k0_27 = inttobitarr(k0, len=32)[31 - 27]

ks1 = k0_21 + k5_15 + k1_6
ks2 = k5_14 + k1_5
ks3 = k4_5 + (k4_4 * k0_27)
k4_29 = inttobitarr(k4, len=32)[31 - 29]
k1_3 = inttobitarr(k1, len=32)[31 - 3]
k1_2 = inttobitarr(k1, len=32)[31 - 2]

print(ks1, ks2, ks3, k4_29, k1_3, k1_2) 
print("I'm stopping here")














 
