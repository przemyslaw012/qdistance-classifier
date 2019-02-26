#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:38:31 2019

@author: Przemyslaw Sadowski

Source code for experiments in arXiv:1803.00853
"""
import numpy as np
from pyquil.quil import Program
from pyquil.gates import H, RY, CNOT


def prepare_3qubit(a, b):
    """
    For a, b being lists of size 8, representing 3-qubit states
    Returns a pyquil Program that prepares 4-qubit state \sqrt(0.5)*(|0>\otimes|a> + |1>\otimes|b>)
    """
    a_ampli = [(np.sqrt(a[2*i]**2+a[2*i+1]**2)) for i in range(4)]
    b_ampli = [(np.sqrt(b[2*i]**2+b[2*i+1]**2)) for i in range(4)]
    assert (abs(sum(np.array(a_ampli)**2)-1)<1e-5)
    assert (abs(sum(np.array(b_ampli)**2)-1)<1e-5)
    preprocessing = prepare_2qubit(a_ampli, b_ampli)

    angles = [np.arccos(a[2*i]/np.sqrt(a[2*i]**2+a[2*i+1]**2)) for i in range(4)]+\
             [np.arccos(b[2 * i] / np.sqrt(b[2 * i] ** 2 + b[2 * i + 1] ** 2)) for i in range(4)]

    for i in range(4):
        if a[2*i+1] < 0:
            angles[i] = -angles[i]
        if b[2*i+1] < 0:
            angles[i+4] = -angles[i+4]

    return preprocessing+controlled_angles_3qubit(angles)


def controlled_angles_3qubit(angles):
    """
    Implements controlled relative phase shift.
    There are 3 control qubits and 8 corresponding phase shifts.
    There is 1 target qubit.
    """
    # preparing8 = [
    #     [ 1, 1, 1, 1, 1, 1, 1, 1],
    #     [-1,-1,-1,-1, 1, 1, 1, 1],
    #     [ 1, 1,-1,-1,-1,-1, 1, 1],
    #     [-1,-1, 1, 1,-1,-1, 1, 1],
    #     [ 1,-1,-1, 1, 1,-1,-1, 1],
    #     [-1, 1, 1,-1, 1,-1,-1, 1],
    #     [ 1,-1, 1,-1,-1, 1,-1, 1],
    #     [-1, 1,-1, 1,-1, 1,-1, 1] ]
    # rotations = np.linalg.solve(preparing8, angles)

    a1, a2, a3, a4, a5, a6, a7, a8 = angles
    rotations = [np.pi/4 + a1 / 8 - a2 / 8 + a3 / 8 - a4 / 8 + a5 / 8 - a6 / 8 + a7 / 8 - a8 / 8,
      a1 / 8 - a2 / 8 + a3 / 8 - a4 / 8 - a5 / 8 + a6 / 8 - a7 / 8 + a8 / 8,
      a1 / 8 - a2 / 8 - a3 / 8 + a4 / 8 - a5 / 8 + a6 / 8 + a7 / 8 - a8 / 8,
      a1 / 8 - a2 / 8 - a3 / 8 + a4 / 8 + a5 / 8 - a6 / 8 - a7 / 8 + a8 / 8,
      a1 / 8 + a2 / 8 - a3 / 8 - a4 / 8 + a5 / 8 + a6 / 8 - a7 / 8 - a8 / 8,
      a1 / 8 + a2 / 8 - a3 / 8 - a4 / 8 - a5 / 8 - a6 / 8 + a7 / 8 + a8 / 8,
      a1 / 8 + a2 / 8 + a3 / 8 + a4 / 8 - a5 / 8 - a6 / 8 - a7 / 8 - a8 / 8,
      a1 / 8 + a2 / 8 + a3 / 8 + a4 / 8 + a5 / 8 + a6 / 8 + a7 / 8 + a8 / 8 - np.pi/4]

    p=Program()
    p.inst(RY(rotations[0]*2, 3))
    p.inst(CNOT(0, 3))
    p.inst(RY(rotations[1]*2, 3))
    p.inst(CNOT(1, 3))
    p.inst(RY(rotations[2]*2, 3))
    p.inst(CNOT(0, 3))
    p.inst(RY(rotations[3]*2, 3))
    p.inst(CNOT(2, 3))
    p.inst(RY(rotations[4]*2, 3))
    p.inst(CNOT(0, 3))
    p.inst(RY(rotations[5]*2, 3))
    p.inst(CNOT(1, 3))
    p.inst(RY(rotations[6]*2, 3))
    p.inst(CNOT(0, 3))
    p.inst(RY(rotations[7]*2, 3))
    return p


def prepare_2qubit(a, b):
    """
    For a, b being lists of size 4, representing 2-qubit states
    Returns a pyquil Program that prepares \sqrt(0.5)*(|0>\otimes|a> + |1>\otimes|b>)
    """
    assert abs(sum([amplitude**2 for amplitude in a])-1)<1e-5
    assert abs(sum([amplitude**2 for amplitude in b])-1)<1e-5
    r1 = np.arccos(np.sqrt(a[0]**2+a[1]**2))
    r2 = np.arccos(np.sqrt(b[0]**2+b[1]**2))
    a1=np.pi/4+(r1-r2)/2
    a2=-np.pi/4+(r1+r2)/2

    p=Program()
    p.inst(H(0))
    p.inst(RY(a1*2, 1))
    p.inst(CNOT(0,1))
    p.inst(RY(a2*2, 1))

    angle1 = np.arccos(a[0]/np.sqrt(a[0]**2+a[1]**2))
    angle2 = np.arccos(a[2]/np.sqrt(a[2]**2+a[3]**2))
    angle3 = np.arccos(b[0]/np.sqrt(b[0]**2+b[1]**2))
    angle4 = np.arccos(b[2]/np.sqrt(b[2]**2+b[3]**2))
    if a[1]<0:
        angle1 = -angle1
    if a[3]<0:
        angle2 = -angle2
    if b[1]<0:
        angle3 = -angle3
    if b[3]<0:
        angle4 = -angle4
    angles = [angle1, angle2, angle3, angle4]
    r3 = (sum(angles[:2])-sum(angles[2:]))/4
    r4 = sum(angles)/4 - np.pi/4
    r1 = (sum(angles[::2]))/2 - sum(angles)/4 + np.pi/4
    r2 = (angles[0]-angles[2])/2 - r3

    p.inst(RY(r1*2, 2))
    p.inst(CNOT(0,2))
    p.inst(RY(r2*2, 2))
    p.inst(CNOT(1,2))
    p.inst(RY(r3*2, 2))
    p.inst(CNOT(0,2))
    p.inst(RY(r4*2, 2))
    return p


if __name__=="__main__":
    help(prepare_3qubit)
    