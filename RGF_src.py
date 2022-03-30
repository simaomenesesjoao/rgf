#!/usr/bin/env python
# coding: utf-8

# # Recursive Green Function 2D - graphene + Rashba + mag - general
# 
# Code to implement RGF<br>
# 

# In[7]:


# to convert to script run
if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script green6_RGF_code.ipynb')


# In[2]:


from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt


# # <b>Hamiltonian:</b> 
# 
# The indexation is: i\*C + j\*No + oo where C is the cross-section (W\*No), No is the number of orbitals 

# ## Graphene nanoribbon

# In[3]:




def ham_gramag(W, hops, twist=False, k=0):
    No = 8
    t,m,R = hops
    C = No*W
    
    sq = np.sqrt(3.0)/2.0
    
    h  = np.zeros([C,C], dtype=complex)
    # print(h)
    for j in range(W):
        
        # up spin
        nAu = 0 + j*No
        nBu = 1 + j*No
        nCu = 2 + j*No
        nDu = 3 + j*No
        
        # down spin
        nAd = 4 + j*No
        nBd = 5 + j*No
        nCd = 6 + j*No
        nDd = 7 + j*No
        

        # print(nAu, nBu, nCu, nDu, nAd, nBd, nCd, nDd)
        # print(No)
        
        # magnetization
        h[nAu, nAu] = m
        h[nBu, nBu] = m
        h[nCu, nCu] = m
        h[nDu, nDu] = m
        
        h[nAd, nAd] = -m
        h[nBd, nBd] = -m
        h[nCd, nCd] = -m
        h[nDd, nDd] = -m
        
        
        # Within same unit cell, spin up
        h[nAu, nBu] = t
        h[nBu, nAu] = t

        h[nCu, nBu] = t
        h[nBu, nCu] = t

        h[nCu, nDu] = t
        h[nDu, nCu] = t
        
        # Within same unit cell, spin down
        h[nAd, nBd] = t
        h[nBd, nAd] = t

        h[nCd, nBd] = t
        h[nBd, nCd] = t

        h[nCd, nDd] = t
        h[nDd, nCd] = t
        
        
        
        
        # Rashba SoC, horizontal, same UC
        h[nBu, nAd] = -R
        h[nAd, nBu] = h[nBu, nAd].conjugate()
        h[nDu, nCd] = -R
        h[nCd, nDu] = h[nDu, nCd].conjugate()
        
        h[nBd, nAu] = -R
        h[nAu, nBd] = h[nBd, nAu].conjugate()
        h[nDd, nCu] = -R
        h[nCu, nDd] = h[nDd, nCu].conjugate()
        
        # Rashba SoC, diagonal, same UC
        h[nBu, nCd] =  -R*(-0.5 + sq*1j)
        h[nCd, nBu] = h[nBu, nCd].conjugate()
        h[nBd, nCu] =  -R*(-0.5 - sq*1j)
        h[nCu, nBd] = h[nBd, nCu].conjugate()
        
        
        
        
        
        
        
        # Connect to upper boundary
        if j<W-1:
            # spin up
            h[nCu, nBu+No] = t
            h[nBu+No, nCu] = t
            
            # spin down
            h[nCd, nBd+No] = t
            h[nBd+No, nCd] = t
            
            # Rashba - different UC
            h[nBu+No, nCd] = -R*(-0.5 - sq*1j)
            h[nCd, nBu+No] = h[nBu+No, nCd].conjugate()

            h[nBd+No, nCu] = -R*(-0.5 + sq*1j)
            h[nCu, nBd+No] = h[nBd+No, nCu].conjugate()
        
        
        
        # Imposing k-point sampling
        if twist and j == W-1:
            # spin up
            h[nCu, 1] = t*np.exp(1j*k)
            h[1, nCu] = t*np.exp(-1j*k)
            
            # spin down
            h[nCd, 5] = t*np.exp(1j*k)
            h[5, nCd] = t*np.exp(-1j*k)
            
            # Rashba - different UC : 1= nBu+No, 5=nBd+No     
            h[1, nCd] = -R*(-0.5 - sq*1j)*np.exp(-1j*k)
            h[nCd, 1] = h[1, nCd].conjugate()

            h[5, nCu] = -R*(-0.5 + sq*1j)*np.exp(-1j*k)
            h[nCu, 5] = h[5, nCu].conjugate()
            
               
    return h

def hop_gramag(W, hops, twist=False, k=0):
    No = 8
    t,m,R = hops
    C = No*W
    sq = np.sqrt(3.0)/2.0
    
    u = np.zeros([C,C], dtype=complex)
    for j in range(W):
        # up spin
        nAu = 0 + j*No
        nBu = 1 + j*No
        nCu = 2 + j*No
        nDu = 3 + j*No
        
        # down spin
        nAd = 4 + j*No
        nBd = 5 + j*No
        nCd = 6 + j*No
        nDd = 7 + j*No
        
        u[nDu,nAu] = t
        u[nDd,nAd] = t
        
        # Rashba SoC
        u[nDu, nAd] = -R*(-0.5 - sq*1j)
        u[nDd, nAu] = -R*(-0.5 + sq*1j)

        
        # Connect to upper boundary
        if j<W-1:
            u[nDu,nAu+No] = t
            u[nDd,nAd+No] = t
            
            # Rashba
            u[nDu, nAd+No] = -R*(-0.5 + sq*1j)
            u[nDd, nAu+No] = -R*(-0.5 - sq*1j)

        # Impose k-point sampling
        if twist and j == W-1:
            u[nDu,0] = t*np.exp(1j*k)
            u[nDd,4] = t*np.exp(1j*k)
            
            # Rashba
            u[nDu, 4] = -R*(-0.5 + sq*1j)*np.exp(1j*k)
            u[nDd, 0] = -R*(-0.5 - sq*1j)*np.exp(1j*k)
            
    return u


# In[4]:


def set_graphene_nanoribbon_rashba(self, width, length, 
                                   twist, k, ander=0.0, m=0.4, lR=0.3):
    # primitive vectors
    a_cc = 1.0
    a1 = np.array([3.0,       0.0])*a_cc
    a2 = np.array([0.0,np.sqrt(3)])*a_cc
    prim = [a1, a2]

    # orbital positions
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])*a_cc
    
    C = B + np.array([1.0, np.sqrt(3)])*a_cc/2
    D = C + np.array([1.0,        0.0])*a_cc
    
    pos = [A*1.0,B*1.0,C*1.0,D*1.0,A*1.0,B*1.0,C*1.0,D*1.0]

    # for i in pos:
    #     print(i)
    #     plt.scatter(i[0], i[1])
    # plt.show()
        
    # Twist parameters
    # twist = True
    # k = 0

    # Geometry parameters
    W = width
    S = length
    No = 8
    C = No*W

    self.W = W
    self.S = S
    self.No = No
    self.C = C
    self.pos = pos
    self.prim = prim
    
    # Hopping parameters
    t = -1
    # m = 0.4
    R = lR*2.0j/3.0
    hops = [t,m,R]
    # ander = 0.0

    

    # Setting the Hamiltonian
    h = ham_gramag(W, hops, twist, k)
    u = hop_gramag(W, hops, twist, k)
    ut = u.transpose().conjugate()
    Anderson = np.random.random([C,S])*ander


    # set the relevant quantities
    self.set_h(h,u)
    self.Anderson = Anderson

    


# ## 2D TB nanoribbon

# In[5]:




def ham_2dtb(W, hops, twist=False, k=0):
    No = 1
    t = hops[0]
    C = No*W
    
    
    h  = np.zeros([C,C], dtype=complex)
    for j in range(W):
        
        
        # Connect to upper boundary
        if j<W-1:
            # spin up
            h[j, j+No] = t
            h[j+No, j] = t
            
        
        
        # Imposing k-point sampling
        if twist and j == W-1:
            # spin up
            h[j, 0] = t*np.exp( 1j*k)
            h[0, j] = t*np.exp(-1j*k)
            
               
    return h

def hop_2dtb(W, hops, twist=False, k=0):
    No = 1
    t = hops[0]
    C = No*W
    
    u = np.zeros([C,C], dtype=complex)
    
    for j in range(W):        
        u[j,j] = t
        
    return u


# In[6]:


def set_2dtb_nanoribbon(self, width, length, twist, k, ander=0.0):
    # primitive vectors
    a_cc = 1.0
    a1 = np.array([1.0, 0.0])*a_cc
    a2 = np.array([0.0, 1.0])*a_cc
    prim = [a1, a2]

    # orbital positions
    A = np.array([0.0, 0.0])
    
    pos = [A*1.0]

#     for i in pos:
#         print(i)
#         plt.scatter(i[0], i[1])
#     plt.show()
        
    # Twist parameters
    # twist = True
    # k = 0

    # Geometry parameters
    W = width
    S = length
    No = 1
    C = No*W

    self.W = W
    self.S = S
    self.No = No
    self.C = C
    self.pos = pos
    self.prim = prim
    
    # Hopping parameters
    t = -1
    hops = [t]
    

    

    # Setting the Hamiltonian
    h = ham_2dtb(W, hops, twist, k)
    u = hop_2dtb(W, hops, twist, k)
    ut = u.transpose().conjugate()
    Anderson = np.random.random([C,S])*ander


    # set the relevant quantities
    self.set_h(h,u)
    self.Anderson = Anderson


# ## TB2D larger UC

# In[7]:



def ham_2dtb_large(W, hops, twist=False, k=0):
    No = 2
    t = hops[0]
    C = No*W
    
    
    h  = np.zeros([C,C], dtype=complex)
    for j in range(W):
        
        A = j*No
        B = A+1
        
        h[A, B] = t
        h[B, A] = t
        
        # Connect to upper boundary
        if j<W-1:
            h[A, A+No] = t
            h[A+No, A] = t
            
            h[B, B+No] = t
            h[B+No, B] = t
            
        
        
        # Imposing k-point sampling
        if twist and j == W-1:
            h[A, 0] = t*np.exp( 1j*k)
            h[0, A] = t*np.exp(-1j*k)
            
            h[B, 1] = t*np.exp( 1j*k)
            h[1, B] = t*np.exp(-1j*k)
            
               
    return h

def hop_2dtb_large(W, hops, twist=False, k=0):
    No = 2
    t = hops[0]
    C = No*W
    
    u = np.zeros([C,C], dtype=complex)
    
    for j in range(W):  
        A = j*No
        B = A+1
        u[B,A] = t
        
        
    return u


# In[8]:


def set_2dtb_nanoribbon_large(self, width, length, twist, k, ander=0.0):
    # primitive vectors
    a_cc = 1.0
    a1 = np.array([2.0, 0.0])*a_cc
    a2 = np.array([0.0, 1.0])*a_cc
    prim = [a1, a2]

    # orbital positions
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    
    pos = [A*1.0, B*1.0]

#     for i in pos:
#         print(i)
#         plt.scatter(i[0], i[1])
#     plt.show()
        
    # Twist parameters
    # twist = True
    # k = 0

    # Geometry parameters
    W = width
    S = length
    No = 2
    C = No*W

    self.W = W
    self.S = S
    self.No = No
    self.C = C
    self.pos = pos
    self.prim = prim
    
    # Hopping parameters
    t = -1
    hops = [t]
    

    

    # Setting the Hamiltonian
    h = ham_2dtb_large(W, hops, twist, k)
    u = hop_2dtb_large(W, hops, twist, k)
    ut = u.transpose().conjugate()
    Anderson = np.random.random([C,S])*ander


    # set the relevant quantities
    self.set_h(h,u)
    self.Anderson = Anderson


# ## Ham utils

# In[9]:


def build_spin(self):
    C = self.C
    S = self.S
    W = self.W
    No = self.No
    Nsites_UC = No//2
    
    self.spinx = np.zeros([C*S,C*S], dtype=complex)
    self.spiny = np.zeros([C*S,C*S], dtype=complex)
    for i in range(S):
        for j in range(W):
            for oo in range(Nsites_UC):
                n = i*C + j*No + oo # spin up
                m = n + Nsites_UC # spin down

                # sy
                self.spiny[n,m] = 1j
                self.spiny[m,n] = -1j

                # sx
                self.spinx[n,m] = 1
                self.spinx[m,n] = 1
    
def build_vels_hsample(self):
    C = self.C
    S = self.S
    W = self.W
    No = self.No
    
    # Sample Hamiltonian within same slice
    H_sample = np.zeros([C*S, C*S], dtype=complex)
    for i in range(S):
        a = i*C
        b = (i+1)*C
        H_sample[a:b, a:b] = self.h
        
        for j in range(C):
            c = a + j
            H_sample[c,c] += self.Anderson[j,i]
            
    
    
    # Sample Hamiltonian within both slices
    for i in range(S-1):
        a = i*C
        b = (i+1)*C
        c = (i+2)*C
        H_sample[a:b,b:c] = self.u
        H_sample[b:c,a:b] = self.ut
    
    
    # position vectors in the sample
    X = np.zeros([C*S, C*S])
    Y = np.zeros([C*S, C*S])
    
    xvec = np.zeros([S,C])
    yvec = np.zeros([S,C])
    
    for i in range(S):
        for j in range(W):
            for o in range(No):
                r = self.prim[0]*i + self.prim[1]*j + self.pos[o]
                m = j*No + o
                n = i*C + j*No + o
                X[n,n] = r[0]
                Y[n,n] = r[1]
                xvec[i,m] = r[0]
                yvec[i,m] = r[1]
              
    self.X = xvec*1.0
    self.Y = yvec*1.0
    
    
    
    # velocity operator
    self.vx = (X@H_sample - H_sample@X)/1.0j
    self.vy = (Y@H_sample - H_sample@Y)/1.0j
    self.H_sample = H_sample
    


# In[ ]:





# In[10]:


def generate_hamiltonian(self, L,R):
    # Generates the full Hamiltonian, including leads. This is mainly for testing purposes
    # Requires the lead sizes L and R

    S = self.S
    C = self.C
    
    NL = L+S+R
    N = NL*C # total Hilbert space
    
    self.NL = NL
    self.N = N
    
    self.H = np.zeros([N,N], dtype=complex)

    # Hops within same cell
    for i in range(NL):
        a = i*C; b = (i+1)*C
        op = 1.0
        if L<=i<L+S:
            op = self.h + np.diag(self.Anderson[:,i-L])
        else:
            op = self.h*1.0

        self.H[a:b,a:b] = op

    # Hop to next slice
    for i in range(NL-1):
        a=i*C;     b=a+C
        c=(i+1)*C; d=c+C

        self.H[a:b,c:d] = self.u*1.0
        self.H[c:d,a:b] = self.ut*1.0


# In[ ]:





# In[ ]:





# # Green's functions with RGF

# ## Lead surface Green's functions

# In[11]:


# @jit(nopython=True)
def build_surface_green_right(self,z, niter=60):
    C = self.C
    matE = np.eye(C)*z
    # print(matE)

    a   = self.u*1.0
    b   = self.ut*1.0
    e1  = self.h*1.0
    e2  = self.h*1.0

    conv = np.zeros([C,C,niter], dtype=complex)


    for i in range(niter):
        g = np.linalg.inv(matE-e2)
        a_new = a@g@a
        b_new = b@g@b
        e1_new = e1 + a@g@b
        e2_new = e2 + a@g@b + b@g@a

        a,b,e1, e2 = a_new, b_new, e1_new, e2_new
        g_surf = np.linalg.inv(matE-e1)
        conv[:,:,i] = g_surf

    return conv

# @jit(nopython=True)
def build_surface_green_left(self,z, niter=60):
    C = self.C
    matE = np.eye(C)*z
    

    a = self.ut*1.0
    b  = self.u*1.0
    e1  = self.h*1.0
    e2  = self.h*1.0

    
    conv = np.zeros([C,C,niter], dtype=complex)


    for i in range(niter):
        g = np.linalg.inv(matE-e2)
        a_new = a@g@a
        b_new = b@g@b
        e1_new = e1 + a@g@b
        e2_new = e2 + a@g@b + b@g@a

        a,b,e1, e2 = a_new, b_new, e1_new, e2_new
        g_surf = np.linalg.inv(matE-e1)
        conv[:,:,i] = g_surf

    return conv

def build_surfaceL(self,zs):

    C = self.C
    assert self.ham_defined == True
        
    NZ = len(zs)
    gL = np.zeros([C,C,NZ], dtype=complex)
    for zz,z in enumerate(zs):
        gL[:,:,zz] = self.build_surface_green_left(z)[:,:,-1]

    return gL

def build_surfaceR(self,zs):

    C = self.C
    assert self.ham_defined == True
        
    NZ = len(zs)
    gR = np.zeros([C,C,NZ], dtype=complex)
    for zz,z in enumerate(zs):
        gR[:,:,zz] = self.build_surface_green_right(z)[:,:,-1]

    return gR

def build_surfaces(self,zs):

    gL = self.build_surfaceL(zs)
    gR = self.build_surfaceR(zs)
    

    return gL, gR


# ## Surface general
# Surface Green's function that is not at the lead

# In[12]:


def build_surface_GL(self, zs, n):
    C = self.C
    S = self.S

    GL_RGF = self.build_surfaceL(zs)*1.0
    
    for i in range(n):
        hi = self.h + np.diag(self.Anderson[:,i])
        for zz, z in enumerate(zs):
            GL_RGF[:,:,zz] = np.linalg.inv(z*np.eye(C)-hi-self.ut@GL_RGF[:,:,zz]@self.u)
            
    return GL_RGF


def build_surface_GR(self, zs, n):
    # up to n+1 inclusive
    C = self.C
    S = self.S
    GR_RGF = self.build_surfaceR(zs)*1.0
    
    # Go from S-1 to n+1 in decreasing order
    # example S=6 and n=2 would be iterated as 5->4->3
    for i in range(S-1,n, -1):
        
        hi = self.h + np.diag(self.Anderson[:,i])
        for zz, z in enumerate(zs):
            GR_RGF[:,:,zz] = np.linalg.inv(z*np.eye(C)-hi-self.u@GR_RGF[:,:,zz]@self.ut)
            
    return GR_RGF


# ## Surface with drop

# In[2]:


def build_surfacedrop_GL(self, zs, n, dV):
    C = self.C
    S = self.S

    GL_RGF = self.build_surfaceL(zs+dV/2)*1.0
    
    length = S*self.prim[0][0]-0.5
    
    for i in range(n):
        pos = self.X[i,:]
        pot = dV/2.0*(1 - 2*pos/length)
        drop = np.diag(pot)
        
        # print(pot)
        hi = self.h + np.diag(self.Anderson[:,i]) + drop
        for zz, z in enumerate(zs):
            GL_RGF[:,:,zz] = np.linalg.inv(z*np.eye(C)-hi-self.ut@GL_RGF[:,:,zz]@self.u)
            
    return GL_RGF


def build_surfacedrop_GR(self, zs, n, dV):
    # up to n+1 inclusive
    C = self.C
    S = self.S
    GR_RGF = self.build_surfaceR(zs-dV/2)*1.0
    
    length = S*self.prim[0][0] - 0.5
    
    
    # Go from S-1 to n+1 in decreasing order
    # example S=6 and n=2 would be iterated as 5->4->3
    for i in range(S-1,n, -1):
        pos = self.X[i,:]
        pot = dV/2.0*(1 - 2*pos/length)
        drop = np.diag(pot)
        # print(pot)
        
        hi = self.h + np.diag(self.Anderson[:,i]) + drop
        for zz, z in enumerate(zs):
            GR_RGF[:,:,zz] = np.linalg.inv(z*np.eye(C)-hi-self.u@GR_RGF[:,:,zz]@self.ut)
            
    return GR_RGF


# ## Local Green

# In[ ]:





# In[13]:



def get_Gnn(self, zs, n):
    
    assert self.ham_defined == True
    C = self.C
    S = self.S
    NZ = len(zs) 

    # Calculate surface Green's functions next to the slice
    GL_RGF = self.build_surface_GL(zs,n)
    GR_RGF = self.build_surface_GR(zs,n)

    # Local Green's function
    Gnn_RGF = np.zeros([C,C,NZ], dtype=complex)
    hi = self.h + np.diag(self.Anderson[:,n])
    for zz, z in enumerate(zs):
        Gnn_RGF[:,:,zz] = np.linalg.inv(z*np.eye(C) - hi - self.ut@GL_RGF[:,:,zz]@self.u - self.u@GR_RGF[:,:,zz]@self.ut)

    return Gnn_RGF


# ## Local Green drop

# In[ ]:


def get_Gnn_drop(self, zs, n, dV):
    
    assert self.ham_defined == True
    C = self.C
    S = self.S
    NZ = len(zs) 

    length = S*self.prim[0][0] - 0.5
    
    # Calculate surface Green's functions next to the slice
    GL_RGF = self.build_surfacedrop_GL(zs,n, dV)
    GR_RGF = self.build_surfacedrop_GR(zs,n, dV)

    # Local Green's function
    
    pos = self.X[n,:]
    pot = dV/2.0*(1 - 2*pos/length)
    drop = np.diag(pot)
    hi = self.h + np.diag(self.Anderson[:,n]) + drop
    
    # print(pot)
    
    Gnn_RGF = np.zeros([C,C,NZ], dtype=complex)
    for zz, z in enumerate(zs):
        Gnn_RGF[:,:,zz] = np.linalg.inv(z*np.eye(C) - hi - self.ut@GL_RGF[:,:,zz]@self.u - self.u@GR_RGF[:,:,zz]@self.ut)

    return Gnn_RGF


# ## G_0n

# In[14]:


def get_G0n(self, zs, n):
    
    assert self.ham_defined == True
    
    S = self.S
    C = self.C
    NZ = len(zs)  
       
    # surface green's function right of slice
    GR_RGF = self.build_surface_GR(zs,n)
    
    # left lead surface Green's function
    gsurf_RGF_L = self.build_surfaceL(zs)

    full_G_LR        = np.zeros([C,C,NZ], dtype=complex)
    full_G_surf_R    = np.zeros([C,C,NZ], dtype=complex)
    partial_G_surf_R = np.zeros([C,C,NZ], dtype=complex)

    # left to n sweep
    for zz, z in enumerate(zs):
        rec = gsurf_RGF_L[:,:,zz]*1.0
        g_0n = gsurf_RGF_L[:,:,zz]*1.0
        for d in range(n):
            hi = self.h + np.diag(self.Anderson[:,d])
            new = np.linalg.inv(z*np.eye(C)-hi-self.ut@rec@self.u)
            g_0n = g_0n@self.u@new
            rec = new*1.0

        partial_G_surf_R[:,:,zz] = rec*1.0
        
        hi = self.h + np.diag(self.Anderson[:,n])
        
        A = self.ut@rec@self.u
        B =  self.u@GR_RGF[:,:,zz]@self.ut
        full_G_surf_R[:,:,zz] = np.linalg.inv(z*np.eye(C) - hi - A - B)
        
        full_G_LR[:,:,zz] = g_0n@self.u@full_G_surf_R[:,:,zz]

    return full_G_LR


# ## G_n,N

# In[15]:


def get_GnNp1(self, zs, n):
    
    assert self.ham_defined == True
    
    S = self.S
    C = self.C
    NZ = len(zs)  

    # Get the right lead surface Green's function
    gsurf_RGF_R = self.build_surfaceR(zs)
       
    # surface green's function left of the slice
    GL_RGF = self.build_surface_GL(zs,n+1)

    full_G_LR        = np.zeros([C,C,NZ], dtype=complex)
    full_G_surf_R    = np.zeros([C,C,NZ], dtype=complex)
    partial_G_surf_R = np.zeros([C,C,NZ], dtype=complex)

    # left to n sweep
    for zz, z in enumerate(zs):
        rec = GL_RGF[:,:,zz]*1.0
        g_0n = GL_RGF[:,:,zz]*1.0
        for d in range(n+1,S):
            hi = self.h + np.diag(self.Anderson[:,d])
            new = np.linalg.inv(z*np.eye(C)-hi-self.ut@rec@self.u)
            g_0n = g_0n@self.u@new
            rec = new*1.0

        partial_G_surf_R[:,:,zz] = rec*1.0
        
        hi = self.h*1.0# + np.diag(self.Anderson[:,S-1])
        
        A = self.ut@rec@self.u
        B =  self.u@gsurf_RGF_R[:,:,zz]@self.ut
        full_G_surf_R[:,:,zz] = np.linalg.inv(z*np.eye(C) - hi - A - B)
        
        full_G_LR[:,:,zz] = g_0n@self.u@full_G_surf_R[:,:,zz]

    return full_G_LR


# ## Local green large

# In[16]:


def get_green_large(self, zs):
    # Get the full Green's function inside the sample
    NE = len(zs)
    

    S = self.S
    C = self.C
    W = self.W
    No = self.No
       

    # Hopping from sample to left lead
    VL = np.zeros([C, C*S], dtype=complex)
    VL[:,:C] = self.u
    VLt = VL.transpose().conjugate()
    
    # Hopping from sample to right lead
    VR = np.zeros([C, C*S], dtype=complex)
    VR[:,C*(S-1):C*S] = self.ut
    VRt = VR.transpose().conjugate()
    
    green = np.zeros([C*S,C*S,NE], dtype=complex)
    
    zs_n = zs.conjugate()
    gsurf_RGF_L_p, gsurf_RGF_R_p = self.build_surfaces(zs)
    
    for zz,z in enumerate(zs):
        ΣL_p = VLt@gsurf_RGF_L_p[:,:,zz]@VL
        ΣR_p = VRt@gsurf_RGF_R_p[:,:,zz]@VR
        
        green[:,:,zz] = np.linalg.inv(z*np.eye(C*S)  - self.H_sample - ΣR_p - ΣL_p)


    return green


# In[ ]:





# # Physical quantities

# ## Landauer

# In[17]:


def get_landauer(self, zs):
    NE = len(zs)
    zs_n = zs.conjugate()
    C = self.C

    # Left to right sweep
    full_G_LR        = np.zeros([C,C,NE], dtype=complex)
    full_G_surf_R    = np.zeros([C,C,NE], dtype=complex)
    partial_G_surf_R = np.zeros([C,C,NE], dtype=complex)

    gsurf_RGF_L,gsurf_RGF_R = self.build_surfaces(zs)
    for zz, z in enumerate(zs):
        rec = gsurf_RGF_L[:,:,zz]*1.0
        g_0n = gsurf_RGF_L[:,:,zz]*1.0
        for d in range(self.S):
            hi = self.h*1.0 + np.diag(self.Anderson[:,d])
            new = np.linalg.inv(z*np.eye(C)-hi-self.ut@rec@self.u)
            g_0n = g_0n@self.u@new
            rec = new*1.0

        partial_G_surf_R[:,:,zz] = rec*1.0

        full_G_surf_R[:,:,zz] = np.linalg.inv(np.linalg.inv(gsurf_RGF_R[:,:,zz]) - self.ut@rec@self.u)
        full_G_LR[:,:,zz] = g_0n@self.u@full_G_surf_R[:,:,zz]



    # Right to left sweep
    partial_G_surf_L = np.zeros([C,C,NE], dtype=complex)
    full_G_RL        = np.zeros([C,C,NE], dtype=complex)
    full_G_surf_L    = np.zeros([C,C,NE], dtype=complex)

    gsurf_RGF_L,gsurf_RGF_R = self.build_surfaces(zs_n)

    for zz, z in enumerate(zs_n):
        rec = gsurf_RGF_R[:,:,zz]*1.0
        g_0n = gsurf_RGF_R[:,:,zz]*1.0
        for d in range(self.S):
            hi = self.h*1.0 + np.diag(self.Anderson[:,self.S - d - 1])
            new = np.linalg.inv(z*np.eye(C)-hi-self.u@rec@self.ut)
            g_0n = g_0n@self.ut@new
            rec = new*1.0

        partial_G_surf_L[:,:,zz] = rec*1.0

        full_G_surf_L[:,:,zz] = np.linalg.inv(np.linalg.inv(gsurf_RGF_L[:,:,zz]) - self.u@rec@self.ut)
        full_G_RL[:,:,zz] = g_0n@self.ut@full_G_surf_L[:,:,zz]



    gsurf_RGF_L_p,gsurf_RGF_R_p = self.build_surfaces(zs)
    gsurf_RGF_L_n,gsurf_RGF_R_n = self.build_surfaces(zs_n)

    tr = np.zeros(NE, dtype=complex)

    for zz,z in enumerate(zs):
        ΣL_p = self.ut@gsurf_RGF_L_p[:,:,zz]@self.u
        ΣR_p =  self.u@gsurf_RGF_R_p[:,:,zz]@self.ut

        ΣL_n = self.ut@gsurf_RGF_L_n[:,:,zz]@self.u
        ΣR_n =  self.u@gsurf_RGF_R_n[:,:,zz]@self.ut

        ΓL = 1j*(ΣL_p - ΣL_n)
        ΓR = 1j*(ΣR_p - ΣR_n)

        op1 = full_G_LR[:,:,zz]
        op2 = full_G_RL[:,:,zz]
        tr[zz] = np.trace(ΓL@op1@ΓR@op2)


    return tr


# ## Kubo-Greenwood

# In[18]:


def kubo_greenwood(self, zs, op1, op2):
    NE = len(zs)
    
    zc = zs.conjugate()

    S = self.S
    C = self.C
    W = self.W
    No = self.No
       
    Gp = self.get_green_large(zs)
    Gn = self.get_green_large(zc)
    
    tr = np.zeros(NE, dtype=complex)
    
    for zz,z in enumerate(zs):
        img = (Gp[:,:,zz] - Gn[:,:,zz])/2j        
        tr[zz] = np.trace(img@op1@img@op2)
        
    return tr


# ## Kubo-Bastin

# In[19]:


def kubo_bastin(self, zs, op, de):
    # returns the integrand of the kubo-Bastin formula which can be used 
    # for several things
    NE = len(zs)
    
    S = self.S
    C = self.C
    W = self.W
    No = self.No    

    # Hopping from sample to left lead
    VL = np.zeros([C, C*S], dtype=complex)
    VL[:,:C] = self.u
    VLt = VL.transpose().conjugate()
    
    # Hopping from sample to right lead
    VR = np.zeros([C, C*S], dtype=complex)
    VR[:,C*(S-1):C*S] = self.ut
    VRt = VR.transpose().conjugate()
    
    tr = np.zeros(NE, dtype=complex)
    
    zs_n = zs.conjugate()
    gz = np.zeros([NE,C*S,C*S], dtype=complex)
    gsurf_RGF_L_p, gsurf_RGF_R_p = self.build_surfaces(zs)
    gsurf_RGF_L_n, gsurf_RGF_R_n = self.build_surfaces(zs_n)
    
    gsurf_RGF_L_p_de, gsurf_RGF_R_p_de = self.build_surfaces(zs + de)
    gsurf_RGF_L_n_de, gsurf_RGF_R_n_de = self.build_surfaces(zs_n + de)
    
    
    for zz,z in enumerate(zs):
        
        ΣL_p = VLt@gsurf_RGF_L_p[:,:,zz]@VL
        ΣR_p = VRt@gsurf_RGF_R_p[:,:,zz]@VR
        
        ΣL_n = VLt@gsurf_RGF_L_n[:,:,zz]@VL
        ΣR_n = VRt@gsurf_RGF_R_n[:,:,zz]@VR
        
        # der
        ΣL_p_de = VLt@gsurf_RGF_L_p_de[:,:,zz]@VL
        ΣR_p_de = VRt@gsurf_RGF_R_p_de[:,:,zz]@VR
        
        ΣL_n_de = VLt@gsurf_RGF_L_n_de[:,:,zz]@VL
        ΣR_n_de = VRt@gsurf_RGF_R_n_de[:,:,zz]@VR


        
        zc = z.conjugate()
        gp = np.linalg.inv(z*np.eye(C*S)  - self.H_sample - ΣR_p - ΣL_p)
        gz[zz,:,:] = gp
        gn = np.linalg.inv(zc*np.eye(C*S) - self.H_sample - ΣR_n - ΣL_n)
        
        dg = (np.linalg.inv((z+de)*np.eye(C*S)  - self.H_sample - ΣR_p_de - ΣL_p_de) - gp)/de
        img = (gp - gn)/2j
        
        # tr[zz] = np.trace(img@vx@dg@spinx)/S/S
        tr[zz] = np.trace(img@self.vx@dg@op)#/S/S


    return tr#,gz


# ## Kubo sea

# In[20]:


def kubo_sea(self, zs, op, de):
    # returns the integrand of the kubo-Bastin formula which can be used 
    # for several things
    NE = len(zs)
    
    S = self.S
    C = self.C
    W = self.W
    No = self.No    

    # Hopping from sample to left lead
    VL = np.zeros([C, C*S], dtype=complex)
    VL[:,:C] = self.u
    VLt = VL.transpose().conjugate()
    
    # Hopping from sample to right lead
    VR = np.zeros([C, C*S], dtype=complex)
    VR[:,C*(S-1):C*S] = self.ut
    VRt = VR.transpose().conjugate()
    
    tr = np.zeros(NE, dtype=complex)
    
    zs_n = zs.conjugate()
    gz = np.zeros([NE,C*S,C*S], dtype=complex)
    gsurf_RGF_L_p, gsurf_RGF_R_p = self.build_surfaces(zs)
    gsurf_RGF_L_n, gsurf_RGF_R_n = self.build_surfaces(zs_n)
    
    gsurf_RGF_L_p_de, gsurf_RGF_R_p_de = self.build_surfaces(zs + de)
    gsurf_RGF_L_n_de, gsurf_RGF_R_n_de = self.build_surfaces(zs_n + de)
    
    
    for zz,z in enumerate(zs):
        
        ΣL_p = VLt@gsurf_RGF_L_p[:,:,zz]@VL
        ΣR_p = VRt@gsurf_RGF_R_p[:,:,zz]@VR
        
        ΣL_n = VLt@gsurf_RGF_L_n[:,:,zz]@VL
        ΣR_n = VRt@gsurf_RGF_R_n[:,:,zz]@VR
        
        # der
        ΣL_p_de = VLt@gsurf_RGF_L_p_de[:,:,zz]@VL
        ΣR_p_de = VRt@gsurf_RGF_R_p_de[:,:,zz]@VR
        
        ΣL_n_de = VLt@gsurf_RGF_L_n_de[:,:,zz]@VL
        ΣR_n_de = VRt@gsurf_RGF_R_n_de[:,:,zz]@VR


        
        zc = z.conjugate()
        gp = np.linalg.inv(z*np.eye(C*S)  - self.H_sample - ΣR_p - ΣL_p)
        gz[zz,:,:] = gp
        gn = np.linalg.inv(zc*np.eye(C*S) - self.H_sample - ΣR_n - ΣL_n)
        
        dgp = (np.linalg.inv((z +de)*np.eye(C*S)  - self.H_sample - ΣR_p_de - ΣL_p_de) - gp)/de
        dgn = (np.linalg.inv((zc+de)*np.eye(C*S)  - self.H_sample - ΣR_n_de - ΣL_n_de) - gn)/de
        img = (gp - gn)/2j
        
        # tr[zz] = np.trace(img@vx@dg@spinx)/S/S
        tr[zz] = np.trace(img@self.vx@(dgp + dgn)@op)


    return tr#,gz


# ## Keldysh

# In[21]:


def keldysh(self, zs,n, op):
    
    NE = len(zs)
    
    S = self.S
    C = self.C
    W = self.W
    No = self.No    
    
    zs_n = zs.conjugate()
    
    tr = np.zeros(NE, dtype=complex)
    
    G0n_adv = self.get_G0n(zs_n,n)
    GnN_ret = self.get_GnNp1(zs,n)
    
    Gnn = self.get_Gnn(zs,n)
    

    
    
    # gz = np.zeros([NE,C*S,C*S], dtype=complex)
    gsurf_RGF_L_p, gsurf_RGF_R_p = self.build_surfaces(zs)
    gsurf_RGF_L_n, gsurf_RGF_R_n = self.build_surfaces(zs_n)
    
    
    tr1 = np.zeros(NE, dtype=complex)
    tr2 = np.zeros(NE, dtype=complex)
    tr3 = np.zeros(NE, dtype=complex)

    for zz,z in enumerate(zs):
        ΣL_p = self.ut@gsurf_RGF_L_p[:,:,zz]@self.u
        ΣR_p =  self.u@gsurf_RGF_R_p[:,:,zz]@self.ut

        ΣL_n = self.ut@gsurf_RGF_L_n[:,:,zz]@self.u
        ΣR_n =  self.u@gsurf_RGF_R_n[:,:,zz]@self.ut

        ΓL = 1j*(ΣL_p - ΣL_n)
        ΓR = 1j*(ΣR_p - ΣR_n)

        op1 = G0n_adv[:,:,zz]
        op2 = op1.transpose().conjugate()
        
        # print(GnN_ret)
        op3 = GnN_ret[:,:,zz]
        op4 = op3.transpose().conjugate()
        
        op5 = Gnn[:,:,zz]
        # op5 = np.imag(Gnn[:,:,zz])
        
        tr1[zz] = np.trace(ΓL@op1@op@op2)
        tr2[zz] = np.trace(op@op3@ΓR@op4)
        tr3[zz] = np.trace(op@op5)

    
    
    return tr1,tr2,tr3


# ## Keldysh drop

# In[ ]:


def keldysh_drop(self, zs,n, op, dV):
    
    NE = len(zs)
    
    # S  = self.S
    # C  = self.C
    # W  = self.W
    # No = self.No    
    
    tr = np.zeros(NE, dtype=complex)
    Gnn = self.get_Gnn_drop(zs,n, dV)

    for zz,z in enumerate(zs):
        
        op5 = Gnn[:,:,zz]
        tr[zz] = np.trace(op@op5)

    return tr


# In[ ]:





# In[ ]:





# In[ ]:





# # Class

# In[22]:




class rgf:
    ham_defined = False
    h = None
    u = None
    ut = None
    
    W  = -1 # sample width
    S  = -1 # sample length
    No = -1 # number of orbitals
    C  = -1 # cross section
    prim = []
    pos = []
    
    vx = -1
    vy = -1
    spinx = -1
    spiny = -1
    H_sample = -1
    H = -1
    
    Anderson = None
    
    def set_h(self, h, u):
        self.h = h*1.0
        self.u = u*1.0
        self.ut = self.u.conjugate().transpose()
        self.ham_defined = True
    
rgf.build_surfaces = build_surfaces
rgf.build_surfaceL = build_surfaceL # has convergence stuff
rgf.build_surfaceR = build_surfaceR # has convergence stuff
rgf.build_surface_green_right = build_surface_green_right # no convergence stuff
rgf.build_surface_green_left = build_surface_green_left # no convergence stuff
rgf.build_surface_GL = build_surface_GL
rgf.build_surface_GR = build_surface_GR

rgf.build_surfacedrop_GR = build_surfacedrop_GR
rgf.build_surfacedrop_GL = build_surfacedrop_GL

rgf.get_green_large = get_green_large
rgf.get_Gnn = get_Gnn
rgf.get_G0n = get_G0n
rgf.get_GnNp1 = get_GnNp1
rgf.get_Gnn_drop = get_Gnn_drop

rgf.get_landauer = get_landauer
rgf.kubo_bastin = kubo_bastin
rgf.kubo_sea = kubo_sea
rgf.kubo_greenwood = kubo_greenwood
rgf.keldysh = keldysh
rgf.keldysh_drop = keldysh_drop

rgf.set_graphene_nanoribbon_rashba = set_graphene_nanoribbon_rashba
rgf.set_2dtb_nanoribbon = set_2dtb_nanoribbon
rgf.set_2dtb_nanoribbon_large = set_2dtb_nanoribbon_large


rgf.build_spin = build_spin
rgf.build_vels_hsample = build_vels_hsample
rgf.generate_hamiltonian = generate_hamiltonian


# In[23]:


def cond(mu, ens, integrand):
    NE = len(ens)
    
    ff = integrand*1.0
    for ee,e in enumerate(ens):
        if e>=mu:
            ff[ee] = 0.0
            
    soma = 0.0
    for ee in range(NE-1):
        dE = ens[ee+1] - ens[ee]
        soma += (ff[ee]+ff[ee+1])/2*dE
    return soma
    

