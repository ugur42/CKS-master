# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:28:46 2023

@author: ugsga

Implementing HHL for section 3.5 of masters paper
"""
import numpy as np
from qiskit import *
from qiskit.circuit.library import QFT

#Define problem parameters
E = 210000
A = 100
F = 10000
L1 = 0.5
L2 = 1
L3 = 1.5
K = E*A*np.array([[1,0,0,0],[0,1/L1+1/L2,-1/L2,0],[0,-1/L2,1/L2+1/L3,-1/L3],[0,0,-1/L3,1/L3]])
r = np.array([[0],[0],[0],[F]])
K = K/np.linalg.norm(K)
r = r/np.linalg.norm(r)

#0 initialize circuit
nb=2
nc=4
na=1
c_reg = QuantumRegister(nc,name="c")
b_reg = QuantumRegister(nb,name="b")
a_reg = QuantumRegister(na,name="a")
classic_reg= ClassicalRegister(3,name="classic")
qc = QuantumCircuit(c_reg,b_reg,a_reg,classic_reg)
qc.barrier(c_reg,b_reg,a_reg,label="0")

#1 initialize b
qc.initialize([0,0,0,1],b_reg)
qc.barrier(c_reg,b_reg,a_reg,label="1")

#2 Hadamard 
for i in range(0,nc):
    qc.h(i)
qc.barrier(c_reg,b_reg,a_reg,label="2")

#3 U=e^-iAt    
U_0 = HamiltonianGate(K,-2**0,label="$U_0$").control() #prepare circuit to implement U
U_1 = HamiltonianGate(K,-2**1,label="$U_1$").control()
U_2 = HamiltonianGate(K,-2**2,label="$U_2$").control()
U_3 = HamiltonianGate(K,-2**3,label="$U_3$").control()
qc.append(U_0,[0,4,5])
qc.append(U_1,[1,4,5])
qc.append(U_2,[2,4,5])
qc.append(U_3,[3,4,5])
qc.barrier(c_reg,b_reg,a_reg,label="3")

#4 IQFT
qc = qc.compose(QFT(4,inverse=True),[0,1,2,3])
qc.barrier(c_reg,b_reg,a_reg,label="4")

#5 controlled Rotation
R0 = CRYGate(2*np.arcsin(1/1)) #theta = 2*np.arcsin(1/c) with |c> = 0001 = 1
R1 = CRYGate(2*np.arcsin(1/2)) #|c> = 0010 = 2
R2 = CRYGate(2*np.arcsin(1/4)) #|c> = 0100 = 4
R3 = CRYGate(2*np.arcsin(1/8)) #|c> = 1000 = 8
qc.append(R0,[0,6])
qc.append(R1,[1,6])
qc.append(R2,[2,6])
qc.append(R3,[3,6])
qc.barrier(c_reg,b_reg,a_reg,label="5")

#6 measure
qc.measure(6,0)
qc.barrier(c_reg,b_reg,a_reg,label="6")

#7 QFT
qc = qc.compose(QFT(4,inverse=False),[0,1,2,3])
qc.barrier(c_reg,b_reg,a_reg,label="7")

#8 U^-1
u_0 = HamiltonianGate(K,2**0,label="$U_0^{-1}$").control() #prepare circuit to implement U
u_1 = HamiltonianGate(K,2**1,label="$U_1^{-1}$").control()
u_2 = HamiltonianGate(K,2**2,label="$U_2^{-1}$").control()
u_3 = HamiltonianGate(K,2**3,label="$U_3^{-1}$").control()
qc.append(u_0,[0,4,5])
qc.append(u_1,[1,4,5])
qc.append(u_2,[2,4,5])
qc.append(u_3,[3,4,5])
qc.barrier(c_reg,b_reg,a_reg,label="8")

#9 Hadamards
for i in range(0,nc):
    qc.h(i)
qc.barrier(c_reg,b_reg,a_reg,label="2")
qc.barrier(c_reg,b_reg,a_reg,label="9")

#10 measure
qc.measure([4,5], [1,2])




circuit_drawer(qc,filename="A_times_A_prime",output='mpl',scale = 0.7,vertical_compression="low")