#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from numpy import linalg as LA


# In[13]:


n=100
x_0=np.zeros((n,1)) #initial point
ave_eps_exact=0
ave_eps_diminish=0
ave_eps_const=0
ave_eps_armijo=0
ave_eps_newton=0
ave_eps_nestrov=0

#print(ave_eps_diminish) 


# In[12]:


for j in range(5):
    B=np.random.randn(n,n)
    D=np.diag(np.exp(np.random.randn(n,1)))
    Q=np.dot(B,B.T)+D
    b=10*np.random.randn(n,1)
    w,c=LA.eig(Q)
    L=np.max(w)
    sigma=np.min((w))
    con_num=L/sigma
    
    #First Order Optimality Condition
    x_opt=-np.dot(np.linalg.inv(Q),b) #optimal X
    opt_func_value=(1/2)*np.dot(np.dot((x_opt).T,Q),x_opt)+np.dot(b.T,x_opt) #optimal value of quadratic function
    
    
    #Exact line search
    X_Exact=x_0
    for i in range(1000):
        grad=np.dot(Q,(X_Exact))+b
        stepsize=np.dot(grad.T,grad)/(np.dot(np.dot(grad.T,Q),grad)) #optmial stepsize for exact line search method
        X_Exact=X_Exact-(stepsize*grad)
        
    epsilon_exact=np.log(np.linalg.norm(X_Exact-x_opt)/np.linalg.norm(x_0-x_opt))
    ave_eps_exact=(ave_eps_exact+epsilon_exact)
    
    
    #Diminishing stepsize rule
    X_Diminish=x_0
    for i in range(1,1001):
        grad=np.dot(Q,X_Diminish)+b
        stepsize=.1/i
        X_Diminish=X_Diminish-stepsize*grad
    
    epsilon_diminish=np.log(np.linalg.norm(X_Diminish-x_opt)/np.linalg.norm(x_0-x_opt))
    ave_eps_diminish=(ave_eps_diminish+epsilon_diminish)
    
    
    #Constant stepsize rule
    X_Const=x_0
    for i in range(1000):
        grad=np.dot(Q,X_Const)+b
        stepsize=1/L
        X_Const=X_Const-stepsize*grad
    epsilon_const=np.log(np.linalg.norm(X_Const-x_opt)/np.linalg.norm(x_0-x_opt))
    ave_eps_const=(ave_eps_const+epsilon_const)


   #Armijo rule

    X_Armijo=x_0

    a=1
    beta=.5
    sigma=.1
    for i in range(0,1000):
        grad=np.dot(Q,X_Armijo)+b
        a=1
        while ((1/2)*np.dot(np.dot(((X_Armijo-a*grad)).T,Q),((X_Armijo-a*grad)))+np.dot(b.T,(X_Armijo-a*grad))-(1/2)*np.dot(np.dot((X_Armijo).T,Q),X_Armijo)-np.dot(b.T,X_Armijo)>a*sigma*np.dot(grad.T,grad)):
            a=a*beta
    
        X_Armijo=X_Armijo-a*grad
       
    epsilon_armijo=np.log(np.linalg.norm(X_Armijo-x_opt)/np.linalg.norm(x_0-x_opt))
    ave_eps_armijo=(ave_eps_armijo+epsilon_armijo)
    
    #Nesterov
    x1=np.zeros((n,1))
    x2=np.zeros((n,1))
    y=np.zeros((n,1))
    a=np.zeros((2000,1))
    t=np.zeros((2000,1))
    for i in range(1,1000):
        a[i+1]=(1/2)*(1+np.sqrt(4*(a[i]**2)+1))
        a[i+2]=(1/2)*(1+np.sqrt(4*(a[i+1]**2)+1))
        t[i+1]=(a[i+1]-1)/a[i+2]
        y=(1+t[i+1])*x2-t[i+1]*x1
        grad=np.dot(Q,y)+b
        x1=x2
        x2=y-(1/L)*grad
        
    X_Nestrov=x2
    epsilon_nesterov=np.log(np.linalg.norm(X_Nestrov-x_opt)/np.linalg.norm(x_0-x_opt))

    ave_eps_nestrov=(ave_eps_nestrov+epsilon_nesterov)
    
    
    #Newton method
    X_Newton=x_0
    W=np.linalg.inv(Q)
    for i in range(1000):
        grad=np.dot(Q,X_Newton)+b
        X_Newton=X_Newton-np.dot(W,grad)

    epsilon_newton=np.log(np.linalg.norm(X_Newton-x_opt)/np.linalg.norm(x_0-x_opt))
    ave_eps_newton=(ave_eps_newton+epsilon_newton)

ave_eps_exact=ave_eps_exact/5
ave_eps_diminish=ave_eps_diminish/5
ave_eps_const=ave_eps_const/5
ave_eps_armijo=ave_eps_armijo/5
ave_eps_newton=ave_eps_newton/5
ave_eps_nestrov=ave_eps_nestrov/5

print(ave_eps_exact)
print(ave_eps_diminish)                                                            
print(ave_eps_const)
print(ave_eps_armijo)   
print(ave_eps_newton)   
print(ave_eps_nestrov)   


# In[ ]:




