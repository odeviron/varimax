#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 07:58:23 2020

@author: odeviron
"""
import numpy as np
def varimax(Phi,  q = 20, tol = 1e-6):
# adapted from wikipedia https://en.wikipedia.org/wiki/Talk%3AVarimax_rotation
# Compute the varimax rotation matrix
# Parameters:
# - Phi: loading 
# - q: max number of iterations
# - tol: convergence treshold
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (1/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)
def run_varimax(data,nm='captvar', cv=50,nmode=0,mv=0.01, **kwargs):
# Compute the PCA varimax decomposition of a 3D matrix along the 2d dimension
# Parameters:
#   data : a 2D numpy array
#   nm (optional): how is fixed the number of modes
#       - 'captvar' (default) percentage of the total variance captured by the sum of the modes
#       - 'minvar' minimum fraction of variance captured by the kept modes
#       - 'nmod'   number of modes
#   cv: percentage of the total variance captured by the sum of the modes (if 'captvar')
#   nmode:   number of modes (if 'nmod')
#   mv: minimum fraction of variance (if 'minvar')
# Returns:
#   ru :  varimax eigen vector
#   rpc: varimax pc
#   rw : varimax captured variance
#
#    
        N,M=data.shape
# Normalization of the input
        for k in range(N):
            st=np.std(data[k,:])
            if st!=0:
                data[k,:]=(data[k,:]-np.mean(data[k,:]))/st
                
# Principal component analysis
        S=np.dot(data.T,data);
        w, v = np.eig(S)
        w=np.real(w)
        v=np.real(v)
        pc=np.dot(np.transpose(v),data.T)
# Mode selection        
        if nm=='captvar':
            s=np.cumsum(w)/sum(w)
            npc=np.nonzero(s>cv*0.01)[0][0]+1
        if nm=='minvar':
            s=w/np.sum(w)
            npc=np.count_nonzero(s>=mv)
        elif nm=='nmod':
            npc=nmode
        w0=np.copy(w)
        ii=np.flip(np.argsort(w))
        w=w[ii]
        pc=pc[ii,:]
        v=v[:,ii]
        w=w[:npc]
        v=v[:,:npc]
        pc=pc[:npc,:]
# call the varimax function
        Lambda=varimax(v,  q = 20, tol = 1e-6)
# construction of the varimax PCs
        ru=Lambda
        rpc=np.dot(ru.T,data.T)
        rw=np.zeros(npc)
        for k in range(npc):
            reck=np.dot(np.reshape(ru[:,k],(M,1)),np.reshape(rpc[k,:],(1,N)))
            rw[k]=1-np.var(data.T-reck)/np.var(data.T)   
        ia=np.flipud(np.argsort(rw))
        rw=rw[ia]
        ru=ru[:,ia]
        rpc=rpc[ia,:] 
        for k in range(npc):
            if ru[:,k].max()<-ru[:,k].min():
                ru[:,k]=-ru[:,k]
                rpc[k,:]=-rpc[k,:]
            if v[:,k].max()<-v[:,k].min():
                v[:,k]=-v[:,k]
                pc[k,:]=-pc[k,:]
        return ru,rpc,rw