# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:08:21 2022

@author: Ugur Gültekin

This file applies the CKS algorithm based on Qiskits HHL algorithm to a
test system
"""
import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from QiskitHHL_to_CKS_2 import CKS
from qiskit.algorithms.linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from qiskit.quantum_info import Statevector
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver

E = 210000
A = 100
F = 10000
L1 = 0.5
L2 = 1
L3 = 1.5

# # 2x2 Matrix:
# K = np.array([[4.,24],[24,1]])
# r = np.array([1,3])
# # print("x_unnorm by np=",np.linalg.solve(A,b))
# norm_K = np.linalg.norm(K)
# norm_r = np.linalg.norm(r)
# K = K/norm_K
# r = r/norm_r

#4x4 Matrix:
K = E*A*np.array([[1,0,0,0],[0,1/L1+1/L2,-1/L2,0],[0,-1/L2,1/L2+1/L3,-1/L3],[0,0,-1/L3,1/L3]])
r = np.array([[0],[0],[0],[F]])

classical_solution = NumPyLinearSolver().solve(K, r / np.linalg.norm(r))
print('classical euclidean norm:', classical_solution.euclidean_norm)
# print('classical state:', classical_solution.state)
# print("expected solution:",np.linalg.solve(K, r))

naive_hhl_solution = CKS().solve(K, r,)
print('naive euclidean norm:', naive_hhl_solution.euclidean_norm)
# print('naive state:')
# print(naive_hhl_solution.state)


#get statevector 
# naive_sv = Statevector(naive_hhl_solution.state).data
# Extract the right vector components. 1000000 corresponds to the index 128 (F register 1 and b regsiter 00)
#, 1000001 corresponds to the index 129  (F register 1 and b regsiter 01)
#, 1000010 corresponds to the index 129  (F register 1 and b regsiter 10)
#, 1000011 corresponds to the index 129  (F register 1 and b regsiter 11)
# naive_full_vector = np.array([naive_sv[128],naive_sv[129],naive_sv[130],naive_sv[131]])
# naive_full_vector = np.real(naive_full_vector)
# print('naive raw solution vector:', naive_full_vector)
# print('full naive solution vector:', naive_hhl_solution.euclidean_norm*naive_full_vector/np.linalg.norm(naive_full_vector))

# print('unnormalised naive solution vector:', naive_hhl_solution.euclidean_norm*naive_full_vector/np.linalg.norm(naive_full_vector)*np.linalg.norm(r))
naive_hhl_solution.state.draw(output='mpl') 
