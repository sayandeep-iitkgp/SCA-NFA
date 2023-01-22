r"""
PRESENT Library for simulating SIFA fault attacks.

AUTHOR: Sayandeep Saha
"""

from random import randint
import fileinput
import collections
import os
import math
import random as pyrandom
from multiprocessing import Process, Queue
import re


from sage.rings.all import FiniteField as GF
from sage.rings.integer_ring import ZZ
from sage.crypto.sbox import SBox
from sage.misc.prandom import random
from sage.combinat.permutation import Permutation
from sage.misc.misc import *



class PRESENT:    
    def __init__(self, Ks=80, Nr=1, B=16, **kwds):
        r"""
        PRESENT is an ultra-lightweight block cipher proposed at CHES
        2007 by A. Bogdanov, L.R. Knudsen , G. Leander , C. Paar,
        A. Poschmann, M.J.B. Robshaw, Y. Seurin, and C. Vikkelsoe

        INPUT:
            B -- blocksize divided by 4 (default: 16)
            Nr -- number of rounds (default: 1)
            Ks -- keysize in (80,128) (default: 80)

        EXAMPLE:
            sage: execfile('present_library_SIFA.py') # output random
            ...
            sage: PRESENT(80,31)
            PRESENT-80-31

        REFERENCES:
            \url{http://www.crypto.rub.de/imperia/md/content/texte/publications/conferences/present_ches2007.pdf}
        """
        if Ks not in (80,128):
            raise ValueError("Number of key bits must be either 80 or 128.")

        if Nr < 1:
            raise ValueError("Number of rounds must be >= 1.")

        self.Ks = Ks
        self.Nr = Nr

        self.B = B
        self.Bs = B*4

        self.s = 4
        self._base = GF(2)
        self._sbox = SBox([12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2], big_endian=True)

    def new_generator(self, **kwds):
        r"""
        Return a new instance of the PRESENT polynomial system
        generator.

        INPUT:
            see constructor

        EXAMPLE:
            sage: execfile('present_library_SIFA.py') #output random
            ...
            sage: p = PRESENT(80,4)
        """
        Nr = kwds.get("Nr",self.Nr)
        B = kwds.get("B",self.B)
        Ks = kwds.get("Ks",self.Ks)
        
        return PRESENT(Ks=Ks, Nr=Nr, B=B)                       

    def __repr__(self):
        r"""
        EXAMPLE:
            sage: execfile('present.py') #output random
            ...
            sage: PRESENT(80,4)
            PRESENT-80-4
        """
        return "PRESENT-%d-%d"%(self.Ks,self.Nr)
    
    def random_element(self, length=None):
        r"""
        Return a random list of elements in $GF(2)$ of the given length

        INPUT:
            length -- length (default: \code{self.Bs})

        EXAMPLE:
            sage: execfile('present.py') # output random
            ... 
            sage: p = PRESENT(Nr=1)
            sage: p.random_element()
            [0, 1, 0, 0, 1, 0, 1, ..., 0, 1, 1, 0, 1, 0, 0, 0, 0]
        """
        if length is None:
            return [self._base.random_element() for _ in range(self.Bs)]
        else:
            return [self._base.random_element() for _ in range(length)]
    
    def gen_random_plaintext(self, N, attack_index=3, pt_file=None, rep=None):
        r"""
        Generate N numbers of random plaintexts 

        INPUT:
        - ``N`` - Number of plaintexts to generate
        - ``pt_file`` - read the inputs from a plaintext file (default: ``None``)
        """
        plaintext_list = []		
        pt_list = []
        if pt_file is not None:
            with open(pt_file) as f_ptr:
                pt_list = [next(f_ptr).rstrip('\n') for i in range(N)]
            for i in pt_list:
                pt = hexstringtobitstring(i)
                plaintext_list.append(pt)
        elif (N == 1):                                   # Default Case
            i = "0011223344556677"
            pt = hexstringtobitstring(i)
            plaintext_list.append(pt)
        elif (N > 1):
            for i in range(N):
                pt = self.random_element()
                plaintext_list.append(pt)
        else:
            raise ValueError('Invalid value for N')

        return plaintext_list

    def gen_ciphertext(self, plaintext_list, key, attack_index=3, config_database=None, faulty=None, fault_model=None, round_index=None, fault_location=None, fno=None, fault_index = None, test_mode = None):
        r"""
        Generate ciphertexts (correct/faulty) for the plaintexts provided in plaintext_list
        
        INPUT:
        - ``plaintext_list`` - List of plaintexts in state_array format
        
        - ``key `` - The key for encryption
        
        - ``faulty`` - Generate faulty ciphertexts (default: ``False``). 
                       Note that if faulty is true fault_model, round_index, 
                       and round_index must be provided whereas, fault_location is optional
        """
        #print(attack_index)
        #pause()
        
        fm = fault_model
        ri = round_index
        fl = fault_location
        
        
        B = self.B
        Bs = self.Bs
        Nbytes = self.Bs/8  	
        
        if faulty is not None:
            if faulty is False:
                faulty = None
            elif faulty is True:
                if ( (fault_model is None) or (round_index is None) ):
                    raise ValueError(' You must specify fault_model and round_index to generate faulty ciphertext')	
                                    
        if ( (faulty is True) and ((fault_model != 1) and (fault_model != 4) and (fault_model != 8)) ):
            raise ValueError('Fault model values must be 1 or 4 or 8')
            
        if ( (faulty is True) and (round_index not in range(1,self.Nr+1)) ):
            raise ValueError('round_index cannot be 0 or more than %d' %self.Nr)
        
        if fault_location is not None: 			
            if ( (faulty is True) and (fault_model == 1) ):
                if ( (fault_location < 0 ) or (fault_location > (Bs -1) ) ):
                    raise ValueError('Fault location out of range')
            elif ( (faulty is True) and (fault_model == 4 ) ):
                if ( (fault_location < 0 ) or (fault_location > (B - 1) ) ):
                    raise ValueError('Fault location out of range')
            elif ( (faulty is True) and (fault_model == 8) ):
                if ( (fault_location < 0 ) or (fault_location > (Nbytes - 1) ) ):
                    raise ValueError('Fault location out of range')

        ciphertext_list = []
        state_container = []
        N = len(plaintext_list)
        for plaintext in plaintext_list:
            if faulty is None:
                tmp1, tmp2 = self(plaintext, key, config_database, attack_index=attack_index, round_index=round_index, test_mode=test_mode)
                ciphertext_list.append(tmp1)
                state_container.append(tmp2)
                #ciphertext_list.append(self(plaintext, key, test_mode))
            else:
                tmp1, tmp2 = self(plaintext, key, config_database, attack_index=attack_index, faulty=True, fault_model=fm, round_index=ri, fault_location=fl,fno=fno, fault_index=fault_index, test_mode=test_mode)
                ciphertext_list.append(tmp1)
                state_container.append(tmp2)
                #ciphertext_list.append(self(plaintext, key, faulty=True, fault_model=fm, round_index=ri, fault_location=fl,fno=fno, test_mode))
                
        return ciphertext_list, state_container

    def cal_hw(self, fno, fault_model):
        fno_bits = map(int, [x for x in '{:0{size}b}'.format(fno,size=fault_model)])
        fno_bits = list(fno_bits)
        hw = 0
        for i in range(len(fno_bits)):
            if (fno_bits[i] == 1):
                hw = hw + 1
        return hw

    def inject_fault(self, S, fault_model, fault_location=None, fno = None ):
        r"""
        Inject the fault in the cipher state.

        INPUT:
        - ``S`` - cipher state
         
        - ``fault_model`` - bit, nibble or byte fault

        - ``fault_location`` - exact location of the fault (default: ``None``)

        """
        B = self.B
        Bs = self.Bs
        Nbytes = self.Bs/8  
        k = self._base
        delta = 0

        # Transition probabilities for simulating ineffective faults
        #-----------------------------------------------------------
        # Stuck-at-0
        #-----------		
        pr_0_0 = 1
        pr_0_1 = 0
        pr_1_0 = 1
        pr_1_1 = 0

        # Random-And
        #------------
        #pr_0_0 = 1
        #pr_0_1 = 0
        #pr_1_0 = 0.5
        #pr_1_1 = 0.5
                
        # Biased bit flip
        #----------------		
        #pr_0_0 = 0.25
        #pr_0_1 = 0.75
        #pr_1_0 = 0.25
        #pr_1_1 = 0.75

        # Random bit flip
        #----------------
        #pr_0_0 = 0.5
        #pr_0_1 = 0.5
        #pr_1_0 = 0.5
        #pr_1_1 = 0.5
                        
        #------------------------------------------------------------		
                                    
        if ( ((fault_model != 1) and (fault_model != 4) and (fault_model != 8)) ):
            raise ValueError('Fault model values must be 1 or 4 or 8')
                    
        if fault_location is not None: 			
            if (fault_model == 1):
                if ( (fault_location < 0 ) or (fault_location > (Bs -1) ) ):
                    raise ValueError('Fault location out of range')
            elif ( fault_model == 4):
                if ( (fault_location < 0 ) or (fault_location > (B - 1) ) ):
                    raise ValueError('Fault location out of range')
            elif ( fault_model == 8):
                if ( (fault_location < 0 ) or (fault_location > (Nbytes - 1) ) ):
                    raise ValueError('Fault location out of range')


        if (fault_model == 1):
            #print(S)
            delta = 1
            S_2 = S
            if fault_location is not None:
                bit_ind = fault_location
                S_2[bit_ind] = S_2[bit_ind] + delta
                S = S_2
            else:
                bit_ind = randint(0,Bs)
                S_2[bit_ind] = S_2[bit_ind] + delta
                S = S_2
        elif (fault_model == 4):
            #delta = self.random_element(length = 4)
            if fno is not None:
                delta = [0,0,0,0]
                f_wd = self.cal_hw(fno, fault_model)	
                nibble_start_ind = 4*fault_location 				
                influenced_bits = pyrandom.sample(range(nibble_start_ind, nibble_start_ind+4), f_wd)
                #print(influenced_bits)
                #pause()
                #influenced_bits = [1,2]
                #influenced_bits = [0,2]
                #influenced_bits = [0]
                for bit_loc in influenced_bits:
                    if (S[bit_loc] == 0):
                        #rnd = sage.misc.prandom.random()
                        rnd = random.random()
                        if (pr_0_1 < rnd):
                            delta[bit_loc - nibble_start_ind] = 0
                        else:
                            delta[bit_loc - nibble_start_ind] = 1
                    else:
                        #rnd = sage.misc.prandom.random()
                        rnd = random.random()
                        if (pr_1_0 < rnd):
                            delta[bit_loc - nibble_start_ind] = 0
                        else:
                            delta[bit_loc - nibble_start_ind] = 1												
                #print(delta)
                #pause()
                """
                # Here we implement in ineffective fault injection mechanism
                #-----------------------------------------------------------				
                # We have made slight modifications to implement the most general fault model,
                # where multiple non-consecutive bits can get affected by faults.
                f_mask = map(int, [x for x in '{:0{size}b}'.format(fno,size=fault_model)])
                #print(f_mask)
                for bit_loc in range(nibble_start_ind, nibble_start_ind+4):
                #for bit_loc in range(nibble_start_ind, nibble_start_ind+f_wd):
                    if (f_mask[(bit_loc - nibble_start_ind)] == 1):
                        if (S[bit_loc] == 0):
                            #rnd = sage.misc.prandom.random()
                            rnd = random()
                            if (pr_0_1 < rnd):
                                delta.append(self._base(0))
                            else:
                                delta.append(self._base(1))
                        else:
                            #rnd = sage.misc.prandom.random()
                            rnd = random()
                            if (pr_1_0 < rnd):
                                delta.append(self._base(0))
                            else:
                                delta.append(self._base(1))
                    else:
                        delta.append(self._base(0))			
                #if ((fault_model - f_wd) > 0):			
                #	delta = delta + [0]*(fault_model - f_wd)
                #print(delta)
                #pause()
                """																					
            else:
                delta = self.random_element(length = 4)
            #while (delta == [0,0,0,0]):
            #	delta = self.random_element(length = 4)
            #	continue
            if fault_location is not None:
                nibble_start_ind = 4*fault_location 
                nibble_end_ind = nibble_start_ind + 3
                target_nibble = S[nibble_start_ind:(nibble_end_ind+1)]
                target_nibble = [ target_nibble[j] + delta[j] for j in range(4) ]
                S[nibble_start_ind:(nibble_end_ind+1)] = target_nibble
            else:
                fault_loc = randint(0,B)
                nibble_start_ind = 4*fault_loc 
                nibble_end_ind = nibble_start_ind + 3
                target_nibble = S[nibble_start_ind:(nibble_end_ind+1)]
                target_nibble = [ target_nibble[j] + delta[j] for j in range(4) ]
                S[nibble_start_ind:(nibble_end_ind+1)] = target_nibble
        else:
            #print(S)
            delta = self.random_element(length = 8)
            while (delta == [0,0,0,0,0,0,0,0]):
                delta = self.random_element(length = 8)
                continue
            if fault_location is not None:
                byte_start_ind = 8*fault_location 
                byte_end_ind = byte_start_ind + 7
                target_byte = S[byte_start_ind:(byte_end_ind+1)]
                target_byte = [ target_byte[j] + delta[j] for j in range(8) ]
                S[byte_start_ind:(byte_end_ind+1)] = target_byte
            else:
                fault_loc = randint(0,Nbytes)
                byte_start_ind = 8*fault_loc 
                byte_end_ind = byte_start_ind + 7
                target_byte = S[byte_start_ind:(byte_end_ind+1)]
                target_byte = [ target_byte[j] + delta[j] for j in range(8) ]
                S[byte_start_ind:(byte_end_ind+1)] = target_byte
                            
        return S	
        
    def sbox(self):
        r"""
        Return SBox object.
        """
        return self._sbox

    def matrix_to_perm(self, matrix):
        r""" converts a permutation matrix to a permutation"""
        perm_list = [None]*len(matrix.column(0))
        dim = len(matrix.column(0))
        for c in range(0,dim):
            col = matrix.column(c)
            for r in range(0,dim):
                if (col[r] == 1):
                    perm_list[c] = r+1				
        return perm_list

    def gen_nibble_perm_to_bit_perm(self, nibble_perm):
        r""" Convert a nibble permutation to a bit permutation"""	
        bit_perm = []
        for i in nibble_perm:
            nibble_start_ind = self.s*i
            for j in range(0,self.s):
                bit_perm.append(self.s*i+j)	
        return bit_perm

    def permute_with_permutation(self, lst, perm):
        r""" Permute a list according to a given permutation"""
        P3 = self.convert_permutation_object_to_present_perm(self.pLayer_as_permutation())
        permute_list = [0]*len(lst)
        for i in range(0,len(lst)):
            permute_list[perm[i]] = lst[i]
            #if (perm == P3):
            #	print(i)
            #	print(permute_list)
            #	pause()
        return permute_list

    def convert_permutation_object_to_present_perm(self, perm_object):
        return [(i-1) for i in list(perm_object)]	

    def convert_present_permutation_to_permutation_object(self, present_perm):
        return Permutation([(i+1) for i in present_perm])	

    def __call__(self, P, K, config_database=None, attack_index=3, faulty=None,fault_model=None,round_index=None,fault_location=None,fno = None, fault_index = None, test_mode= None):

        B = self.B
        Bs = self.Bs
        Nbytes = self.Bs/8  
        k = self._base


        if faulty is not None:
            if faulty is False:
                faulty = None
            elif faulty is True:
                if ( (fault_model is None) or (round_index is None) ):
                    raise ValueError(' You must specify fault_model and round_index to generate faulty ciphertext')	
                                    
        if ( (faulty is True) and ((fault_model != 1) and (fault_model != 4) and (fault_model != 8)) ):
            raise ValueError('Fault model values must be 1 or 4 or 8')
            
        if ( (faulty is True) and (round_index not in range(1,self.Nr+1)) ):
            raise ValueError('round_index cannot be 0 or more than %d' %self.Nr)
                    
        if fault_location is not None:
            if ( (faulty is True) and (fault_model == 1) ):
                if ( (fault_location < 0 ) or (fault_location > (Bs -1) ) ):
                    raise ValueError('Fault location out of range')
            elif ( (faulty is True) and (fault_model == 4) ):
                if ( (fault_location < 0 ) or (fault_location > (B - 1) ) ):
                    raise ValueError('Fault location out of range')
            elif ( (faulty is True) and (fault_model == 8) ):
                if ( (fault_location < 0 ) or (fault_location > (Nbytes - 1) ) ):
                    raise ValueError('Fault location out of range')
                        
        #print(attack_index)
        #pause()
        
        Zi = [ self._base(e) for e in P ]
        state_container = []

        if (attack_index == 1):
            #### Actual PRESENT Code
            ########################
            for i in range(1,self.Nr+1):
                if get_verbose() > 1:
                    reg = bitstringtohexstring(K)
                Ki,K = self.keySchedule(K, i)
                # Inject the fault------------------------------------ 
                if faulty is True:
                    if ( i ==  round_index):
                        if fault_location is not None:
                            Zi = self.inject_fault(Zi,fault_model,fault_location,fno=fno)
                        else:
                            Zi = self.inject_fault(Zi,fault_model)
                #-----------------------------------------------------
                Xi = self.addRoundKey( Zi, Ki )
                #if(i == 28):
                #	print(bitstringtohexstring(Xi))
                if test_mode is True:
                    if(i >= round_index):
                        state_container.append(Xi)
                Yi = self.sBoxLayer( Xi )
                # Inject the fault------------------------------------ 
                #if faulty is True:
                #   if ( i ==  round_index):
                #       if fault_location is not None:
                #           Yi = self.inject_fault(Yi,fault_model,fault_location,fno=fno)
                #       else:
                #           Yi = self.inject_fault(Yi,fault_model)
                #-----------------------------------------------------
                Zi = self.pLayer(Yi)
                #if faulty is True:
                #   if(i == 30):
                #       print(bitstringtohexstring(Yi))
                
            Ki,K = self.keySchedule(K, self.Nr+1)
            #if faulty is True:
            #	print(bitstringtohexstring(Zi))
            Xi = self.addRoundKey(Zi, Ki)
            #pause()
            if test_mode is True:
                state_container.append(Xi)       
        elif ( (attack_index == 2) or (attack_index == 3)):
                
            #### PRESENT Threshold Implementation (3 share)
            # (key is not shared)
            ##################################################################
                            
            # Generate two random masks of 64 bit each
            M1 = self.random_element(length=self.Bs)
            M2 = self.random_element(length=self.Bs)

            #print(bitstringtohexstring(M1))
            #print(bitstringtohexstring(M2))

            # Generate shares
            Zi1 = [Zi[i] + M1[i] + M2[i] for i in range(self.Bs)]
            Zi2 = M1
            Zi3 = M2
            for i in range(1,self.Nr+1):
                # Inject the fault------------------------------------ 
                if (attack_index == 2):
                    if faulty is True:
                        if ( i ==  round_index):
                            if fault_location is not None:
                                #Zi = self.inject_fault(Zi,fault_model,fault_location,fno=fno)
                                Zi1 = self.inject_fault(Zi1,fault_model,fault_location,fno=fno)
                            else:
                                #Zi = self.inject_fault(Zi,fault_model)
                                Zi1 = self.inject_fault(Zi1,fault_model)
                #-----------------------------------------------------				
                # Addroundkey
                Ki,K = self.keySchedule(K, i)
                Xi1 =  self.addRoundKey_threshold(Zi1, Ki)
                Xi2 = Zi2
                Xi3 = Zi3		
                
                # Print intermediate round (for verification purpose)
                if faulty is None:
                    if (i == round_index):
                        Xi_test = [Xi1[j] + Xi2[j] + Xi3[j] for j in range(self.Bs)]
                        #print(bitstringtohexstring(Xi_test))
                
                # sBoxLayer
                Yi1, Yi2, Yi3 = self.sBoxLayer_threshold(Xi1, Xi2, Xi3)
                # Inject the fault------------------------------------ 
                if (attack_index == 3):
                    if faulty is True:
                        if ( i ==  round_index):
                            Yi1, Yi2, Yi3 = self.sBoxLayer_threshold(Xi1, Xi2, Xi3, fault_location, fault_index)
                        else:
                            Yi1, Yi2, Yi3 = self.sBoxLayer_threshold(Xi1, Xi2, Xi3)
                    else:
                        Yi1, Yi2, Yi3 = self.sBoxLayer_threshold(Xi1, Xi2, Xi3)	
                    
                #pLayer
                Zi1, Zi2, Zi3 = self.pLayer_threshold(Yi1, Yi2, Yi3)

            Ki,K = self.keySchedule(K, self.Nr+1)
            Xi1 =  self.addRoundKey_threshold(Zi1, Ki)
            Xi2 = Zi2
            Xi3 = Zi3

            Xi = [Xi1[j] + Xi2[j] + Xi3[j] for j in range(self.Bs)] # Combine the masks to get actual output ciphertext
        else:
            raise ValueError(' Wrong value for attack_index. Correct values \{1, 2, 3\}')
        return Xi, state_container

    def three_shared_G(self, X1, X2, X3):
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

        one = self._base(1)

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

    def three_shared_F(self, X1, X2, X3):	
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

    def three_share_sbox(self, X1, X2, X3):
        G1, G2, G3 = self.three_shared_G(X1, X2, X3)
        F1, F2, F3 = self.three_shared_F(G1, G2, G3)
        return F1, F2, F3

    def three_shared_G_faulted(self, X1, X2, X3, fault_index = 0):

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

        one = self._base(1)


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

        if (fault_index == 0):
            x3_f = 1
            g10 = one + w2 + x2*y2 + x2*y3 + x3_f*y2 + x2*z2 + x2*z3 + x3_f*z2 + y2*z2 + y2*z3 + y3*z2
            g20 = w3 + x3_f*y3 + x1*y3 + x3_f*y1 + x3_f*z3 + x1*z3 + x3_f*z1 + y3*z3 + y1*z3 + y3*z1
            g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1
        elif (fault_index == 1):
            y3_f = 1
            g10 = one + w2 + x2*y2 + x2*y3_f + x3*y2 + x2*z2 + x2*z3 + x3*z2 + y2*z2 + y2*z3 + y3_f*z2
            g20 = w3 + x3*y3_f + x1*y3_f + x3*y1 + x3*z3 + x1*z3 + x3*z1 + y3_f*z3 + y1*z3 + y3_f*z1
            g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1
        elif (fault_index == 2):
            z3_f = 1
            g10 = one + w2 + x2*y2 + x2*y3 + x3*y2 + x2*z2 + x2*z3_f + x3*z2 + y2*z2 + y2*z3_f + y3*z2
            g20 = w3 + x3*y3 + x1*y3 + x3*y1 + x3*z3_f + x1*z3_f + x3*z1 + y3*z3_f + y1*z3_f + y3*z1
            g30 = w1 + x1*y1 + x1*y2 + x2*y1 + x1*z1 + x1*z2 + x2*z1 + y1*z1 + y1*z2 + y2*z1		
        elif (fault_index == 3):
            y3_f = 1
            g11 = one + x2 + z2 + y2*w2 + y2*w3 + y3_f*w2 + z2*w2 + z2*w3 + z3*w2
            g21 = x3 + z3 + y3_f*w3 + y1*w3 + y3_f*w1 + z3*w3 + z1*w3 + z3*w1
            g31 = x1 + z1 + y1*w1 + y1*w2 + y2*w1 + z1*w1 + z1*w2 + z2*w1
        else:
            print("Error!!")			

        return [g13, g12, g11, g10], [g23, g22, g21, g20], [g33, g32, g31, g30]

    def three_shared_F_faulted(self, X1, X2, X3, fault_index = 0):	
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

    def three_share_sbox_faulted(self, X1, X2, X3, fault_index = 0):
        G1, G2, G3 = self.three_shared_G_faulted(X1, X2, X3, fault_index)
        F1, F2, F3 = self.three_shared_F_faulted(G1, G2, G3, fault_index)
        return F1, F2, F3

    def addRoundKey_threshold(self, X, K):
        return [ X[j] + K[j] for j in range(self.Bs) ]

    def addRoundKey_threshold_red(self, X, K, d=1):
        return [ X[j] + K[j] for j in range((2*d+1)*self.Bs) ]

    def sBoxLayer_threshold(self, X1, X2, X3, fault_location = None, fault_index = None):
        s = self.s
        Bs = self.Bs
        Y1 = []
        Y2 = []
        Y3 = []
        byte_loc = 0 
        for i in range(0,Bs,s):
            x1 = X1[i:i+s]
            x2 = X2[i:i+s]
            x3 = X3[i:i+s]
            if ( (fault_location is None) or (fault_index is None) ):
                y1, y2, y3 = self.three_share_sbox(x1, x2, x3)
            else:
                if (fault_location == byte_loc):
                    y1, y2, y3 = self.three_share_sbox_faulted(x1, x2, x3, fault_index)
                else:
                    y1, y2, y3 = self.three_share_sbox(x1, x2, x3)		
            
            Y1 = Y1 + y1
            Y2 = Y2 + y2
            Y3 = Y3 + y3
            byte_loc = byte_loc + 1
            
        return Y1, Y2, Y3

    def pLayer_threshold(self, X1, X2, X3):
        B = self.B
        s = self.s
        Y1 = [0]*B*s
        Y2 = [0]*B*s
        Y3 = [0]*B*s

        Perm_actual = self.pLayer(range(0,self.Bs))
        for i in range(0, B*s):
            Y1[i] = X1[Perm_actual[i]]
            Y2[i] = X2[Perm_actual[i]]
            Y3[i] = X3[Perm_actual[i]]
                    
        return Y1, Y2, Y3

    def decrypt(self, C, K):
        r"""
        Encrypt plaintext P with key K.

        INPUT:
            C -- ciphertext
            K -- key
        """
        Zi = [ self._base(e) for e in C ]

        KK = []

        for i in range(1,self.Nr+1):
            k,K = self.keySchedule(K, i)
            KK.append(k)
        k,K = self.keySchedule(K, self.Nr+1)
        KK.append(k)
        
        for i in reversed(range(1,self.Nr+1)):
            Xi = self.addRoundKey( Zi, KK[i] )
            Yi = self.pLayer_inverse(Xi)
            Zi = self.sBoxLayer_inverse( Yi )
        Xi = self.addRoundKey(Zi, KK[0])
        
        return Xi

    def round_counter(self, i):
        r"""
        INPUT:
            i -- integer

        EXAMPLE:
            sage: execfile('present_library_SIFA.py') # output random
            ... 
            sage: p = PRESENT(80,31)
            sage: for i in range(1,31+1):
            ...     p.round_counter(i)
            ...
            [0, 0, 0, 0, 1]
            [0, 0, 0, 1, 0]
            [0, 0, 0, 1, 1]
            [0, 0, 1, 0, 0]
            [0, 0, 1, 0, 1]
            [0, 0, 1, 1, 0]
            [0, 0, 1, 1, 1]
            [0, 1, 0, 0, 0]
            [0, 1, 0, 0, 1]
            [0, 1, 0, 1, 0]
            [0, 1, 0, 1, 1]
            [0, 1, 1, 0, 0]
            [0, 1, 1, 0, 1]
            [0, 1, 1, 1, 0]
            [0, 1, 1, 1, 1]
            [1, 0, 0, 0, 0]
            [1, 0, 0, 0, 1]
            [1, 0, 0, 1, 0]
            [1, 0, 0, 1, 1]
            [1, 0, 1, 0, 0]
            [1, 0, 1, 0, 1]
            [1, 0, 1, 1, 0]
            [1, 0, 1, 1, 1]
            [1, 1, 0, 0, 0]
            [1, 1, 0, 0, 1]
            [1, 1, 0, 1, 0]
            [1, 1, 0, 1, 1]
            [1, 1, 1, 0, 0]
            [1, 1, 1, 0, 1]
            [1, 1, 1, 1, 0]
            [1, 1, 1, 1, 1]
        """
        rc = list(reversed(ZZ(i).digits(base=2)))
        if len(rc) < 5:
            rc = [0]*(5-len(rc)) + rc

        rc = map(self._base, rc[-5:])
        rc = list(rc)
        return rc

    def keySchedule(self, K, i):
        r"""
        INPUT:
            K -- key register of size self.Ks
            i -- round counter
        """
        S = self.sbox()
        Bs = self.Bs
        Ki = [0]*Bs

        # extract key
        for j in range(Bs):
            Ki[j] = K[j]

        if self.Ks == 80:
            # update register
            K = K[61:80] + K[0:61]
            K[0:4] = S(K[0:4])

            # add round counter
            rc = self.round_counter(i)

            K[80-1-19] += rc[0]
            K[80-1-18] += rc[1]
            K[80-1-17] += rc[2]
            K[80-1-16] += rc[3]
            K[80-1-15] += rc[4]
            return Ki, K

        elif self.Ks == 128:
            # update register
            K = K[61:128] + K[0:61]
            K[0:4] = S(K[0:4])
            K[4:8] = S(K[4:8])

            # add round constant
            rc = self.round_counter(i)

            K[128-1-66] += rc[0]
            K[128-1-65] += rc[1]
            K[128-1-64] += rc[2]
            K[128-1-63] += rc[3]
            K[128-1-62] += rc[4]
            return Ki, K
            
    def addRoundKey(self, X, Y):
        r"""
        Return list of pairwise sums of elements in X and Y.

        INPUT:
            X -- list
            Y -- list
        """
        return [ X[j] + Y[j] for j in range(self.Bs) ]

    def sBoxLayer(self, X):
        r"""
        Apply S-boxes to X
        """
        s = self.s
        sbox = self.sbox()
        return sum([ sbox(X[j:j+s]) for j in range(0,self.Bs,s) ],[])

    def sBoxLayer_serial(self, X):
        r"""
        Apply S-boxes to X
        """
        s = self.s
        sbox = self.sbox()
        Y = []
        sbox_table = [12,5,6,11,9,0,10,13,3,14,15,8,4,7,1,2]
        for j in range(0,self.Bs,s):
            t1 = sbox_table[int(bitstringtohexstring(X[j:j+s]), 16)]
            Y.append(t1)
        Y = hexstringtobitstring(intarraytohexstring(Y))		
        #return sum([ sbox(X[j:j+s]) for j in range(0,self.Bs,s) ],[])
        return Y

    def sBoxLayer_inverse(self, X):
        r"""
        Apply inverse of S-boxes to X
        """
        s = self.s
        sbox = self.sbox()
        inv_sbox = [0 for _ in range(2**s)]
        for i in range(2**s):
            inv_sbox[sbox[i]] = i
        #inv_sbox = mq.SBox(inv_sbox)
        inv_sbox = SBox(inv_sbox)
        return sum([ inv_sbox(X[j:j+s]) for j in range(0,self.Bs,s) ],[])

    def pLayer(self, Y):
        r"""
        Return a list of length self.Bs with linear combinations of
        the elements in the list y (of length self.Bs) matching the
        permutation layer of PRESENT.

        INPUT:
            Y -- list of length self.Bs
        """
        B = self.B
        s = self.s
        Z = [0]*B*s
        for i in range(B):
            for j in range(s):
                Z[B*j + i] = Y[s*i + j] 
                
        return Z
        
    def pLayer_inverse(self, Y):
        r"""
        Return a list of length self.Bs with linear combinations of
        the elements in the list y (of length self.Bs) matching the
        inverse of the permutation layer of PRESENT.

        INPUT:
            Y -- list of length self.Bs
        """
        B = self.B
        s = self.s
        Z = [0]*B*s
        for i in range(B):
            for j in range(s):
                Z[s*i + j] = Y[B*j + i] 
                
        return Z

    def pLayer_as_permutation(self):
        r"""
        """
        return Permutation( self.pLayer(range(1,65)) )


def bitstringtohexstring(l):
    r"""
    Return a hex string in PRESENT style for l a list of bits.

    INPUT:
        l -- a list with bit entries of length divisible by 4
    """
    r = []
    for i in range(0,len(l),4):
        kk = map(int, l[i:i+4])
        kk = list(kk)
        z = list(reversed(kk))
        r.append(format(ZZ(z,2), 'x'))

    r = sum([r[i:i+8]+[" "] for i in range(0,len(r),8) ],[])

    return "".join(r)[:-1]

def hexstringtobitstring(n, length=64):
    r"""
    Return a hex string in PRESENT style for l a list of bits.

    INPUT:
        l -- a list with bit entries of length divisible by 4
    """
    n = int(n,16)
    l = []
    for i in range(length):
        l.append(1 if 2**i & n else 0)
    l = map(GF(2),l)
    l = list(l)
    return list(reversed(l))

def intarraytohexstring(l):
    r"""
    Return a hex string for l a list of integers in [0,15].

    INPUT:
        l -- a list with bit entries of length divisible by 4
    """	
    hexstr = ''.join('{:0x}'.format(x) for x in l)
    return hexstr

def bitstringtointarray(l):
    return [int(t, 16) for t in bitstringtohexstring(l).replace(" ", "")]	

def pause():
    programPause = input("Press the <ENTER> key to continue...")
