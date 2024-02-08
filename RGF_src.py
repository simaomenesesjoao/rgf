#!/usr/bin/env python
# coding: utf-8

# # <font size=7> Recursive Green Function 2D </font> 
# 
# Code to implement RGF in 2D. This notebook is used to produce a python script which contains the library

# In[12]:


# to convert to script run
if __name__== "__main__":
    get_ipython().system('jupyter-nbconvert --to script RGF_src.ipynb')


# In[10]:


from multiprocessing import Pool
import numpy as np
import sys

sys.path.append('../tight-binding-test/src/')
sys.path.append('tight-binding-test/src/')
    
import band_structure as bs


# # Hamiltonian
# 
# This section contains:
# 1. Functions to define the Hamiltonian of the system. This code uses the recursive Green's function method, so the system is assumed to be a nanoribbon (sample + leads) and so has to be defined in terms of slices. The Hamiltonian has to be specified in terms of hoppings within the same slice ($h$) or to the next slice ($u$). 
# 2. Functions to define the velocity and spin operators from this Hamiltonian. Be careful with the basis ordering when dealing with spin.
# 3. Functions that build a finite Hamiltonian, which explicitly includes finite leads. This is for testing purposes
# 
# The indexation is: i\*C + j\*No + oo where C is the cross-section (W\*No), No is the number of orbitals. The term 'orbitals' encompasses spin

# ## Define generic Hamiltonian in the nanoribbon from arbitrary hoppings

# In[3]:


def hamiltonian_UC(W, hops, orbs_dic, twist, k):
    
    No = len(orbs_dic.keys())
        
    C = W*No
    h = np.zeros([C,C], dtype=complex)
    u = np.zeros([C,C], dtype=complex)

    for hop in hops:

        o1,o2,n,m,t = hop
        o1i = orbs_dic[o1]
        o2i = orbs_dic[o2]

        # print(hop, o1i,o2i)
        for w in range(W):

            # Decide the boundary conditions for the sites in the margins
            ph = 1 # hopping phase term
            if w == W-1 and m == 1 or w == 0 and m == -1:

                # Twisted boundary conditions
                if twist:
                    ph = np.exp(-1j*k*m)

                # Open boundary conditions
                else:
                    ph = 0


            i = o1i + w*No
            j = o2i + ((w+m)%W)*No
            # print(i,j)
            if n == 0: h[j,i] += t*ph
            if n == 1: u[j,i] += t*ph 
            
    return h,u.transpose().conjugate()



# In[4]:


def set_system(self, ham_struct, width, length, twist, k):
    
    orbs_dic, pos, prim, hops = ham_struct
    self.orbs_dic = orbs_dic
    self.hops = hops
    self.pos = pos
    self.prim = prim
    
    No = len(orbs_dic.keys())
    
    self.drop_length = 3.0*length - 0.5 

    self.W = width
    self.S = length
    self.No = No
    self.C = No*width
    
    
    # Information about the Hamiltonian
    h,u = hamiltonian_UC(width, hops, orbs_dic, twist, k)

    # set the relevant quantities
    self.set_h(h,u)
    # self.Anderson = Anderson


# ## Velocity and spin operators
# 
# Define the velocity and spin operators inside the whole sample. This assumes infinite leads because it uses the $u$ and $h$ operators
# 
# 1. Build the velocity operators in the $x$ and $y$ directions
# 2. Build the spin operators $s_x$, $s_y$, $s_z$ with two different basis orderings. The first ordering indexes the basis atoms first, then the spin. The second ordering is the other way around
# 

# In[5]:


def build_spin(self):
    # Assumes spin
    C  = self.C
    S  = self.S
    W  = self.W
    No = self.No
    Nsites_UC = No//2
    
    self.spinx = np.zeros([C*S,C*S], dtype=complex)
    self.spiny = np.zeros([C*S,C*S], dtype=complex)
    self.spinz = np.zeros([C*S,C*S], dtype=complex)
    for i in range(S):
        for j in range(W):
            for oo in range(Nsites_UC):
                n = i*C + j*No + oo # spin up
                m = n + Nsites_UC   # spin down

                # sy
                self.spiny[n,m] = 1j
                self.spiny[m,n] = -1j

                # sx
                self.spinx[n,m] = 1
                self.spinx[m,n] = 1
                
                # sz
                self.spinz[n,n] =  1
                self.spinz[m,m] = -1
    
    
def build_spin2(self):
    # Assumes spin, but with a different basis ordering
    C  = self.C
    S  = self.S
    W  = self.W
    No = self.No
    Nsites_UC = No//2
    
    self.spinx = np.zeros([C*S,C*S], dtype=complex)
    self.spiny = np.zeros([C*S,C*S], dtype=complex)
    self.spinz = np.zeros([C*S,C*S], dtype=complex)
    for i in range(S):
        for j in range(W):
            for oo in range(Nsites_UC):
                n = i*C + j*No + 2*oo # spin up
                m = n + 1 # spin down

                # sy
                self.spiny[n,m] = 1j
                self.spiny[m,n] = -1j

                # sx
                self.spinx[n,m] = 1
                self.spinx[m,n] = 1
                
                # sz
                self.spinz[n,n] =  1
                self.spinz[m,m] = -1
                
      


# In[6]:


def build_vels_hsample(self):
    C = self.C
    S = self.S
    W = self.W
    No = self.No
    
    H_sample = np.zeros([C*S, C*S], dtype=complex)
    
    # Sample Hamiltonian within same slice
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
        H_sample[b:c,a:b] = self.ut
        H_sample[a:b,b:c] = self.u
        
    
    
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
                n = i*C + m
                X[n,n] = r[0]
                Y[n,n] = r[1]
                xvec[i,m] = r[0]
                yvec[i,m] = r[1]
              
    self.X = xvec*1.0
    self.Y = yvec*1.0
    
    
    
    # velocity operator
    self.vx = (X@H_sample - H_sample@X)/1.0j
    self.vy = (Y@H_sample - H_sample@Y)/1.0j
    self.H_sample = H_sample*1.0
    


# In[ ]:





# ## Finite leads
# 
# Additional functionality to include leads explicitly inside the full Hamiltonian. This is mostly for testing purposes.
# 1. Generate the Hamiltonian without the drop in potential
# 2. Add the potential drop to the Hamiltonian
# 3. Get the eigenvalues and eigenvectors of this full Hamiltonian

# In[7]:


def generate_hamiltonian(self, L,R):
    # Generates the full Hamiltonian H0, including leads, without drop
    # This is mainly for testing purposes
    # Requires the lead sizes L and R

    S = self.S
    C = self.C
    self.L = L
    self.R = R
    
    NL = L+S+R
    N = NL*C # total Hilbert space
    
    self.NL = NL
    self.N = N
    
    self.H0 = np.zeros([N,N], dtype=complex)

    # Hops within same cell
    for i in range(NL):
        a = i*C; b = (i+1)*C
        op = 1.0
        if L<=i<L+S:
            op = self.h + np.diag(self.Anderson[:,i-L])
        else:
            op = self.h*1.0

        self.H0[a:b,a:b] = op

    # Hop to next slice
    for i in range(NL-1):
        a=i*C;     b=a+C
        c=(i+1)*C; d=c+C

        self.H0[a:b,c:d] = self.u*1.0
        self.H0[c:d,a:b] = self.ut*1.0
    
    self.H0_finite_defined = True


# In[8]:


def hamiltonian_add_drop(self, dV):
    # Include the potential drop in the existing Hamiltonian
    assert(self.H0_finite_defined) # make sure it's defined first

    NL = self.NL
    W  = self.W
    C  = self.C
    L  = self.L
    R  = self.R
    S  = self.S
    No = self.No
    self.dV = dV
    N = NL*C # total Hilbert space
    
    xmin = np.min(self.X)
    xmax = np.max(self.X)
    size = xmax - xmin  # extent of the drop

    #print(xmin, xmax)

    Xp = (self.X - xmin)/size

    drop = np.zeros([NL, C])
    drop[:L,:] = dV/2
    drop[L+S:,:] = -dV/2
    drop[L:L+S,:] = dV*(1.0-2*Xp)/2


    # Drop matrix defined in the whole Hilbert space
    dropmat = np.zeros([N,N], dtype=complex)
    for i in range(NL):
        for j in range(W):
            for oo in range(No):
                m = j*No + oo
                n = i*C + m
                dropmat[n,n] = drop[i,m]

    # Add the potential drop to the Hamiltonian
    self.H = self.H0 + dropmat
    self.H_finite_defined = True


# In[9]:


def get_eigs(self):
    # Get the eigenvalues and eigenvectors of the Hamiltonian with and without drop
    assert(self.H0_finite_defined)
    assert(self.H_finite_defined)
    
    self.vals, self.P = np.linalg.eigh(self.H)
    self.Pt = self.P.conjugate().transpose()

    self.vals0, self.P0 = np.linalg.eigh(self.H0)
    self.Pt0 = self.P0.conjugate().transpose()


# # Premade models 
# These functions are an interface to the bandstructure code
# 1. Graphene nanoribbon with Rashba (predefined hops, no Anderson)
# 2. Twisted TMD/Graphene (Rashba angle), for the project with Klaus and Jaroslav
# 3. Graphene nanoribbon with Rashba and exchange coupling (with Anderson, can choose all couplings)

# In[10]:


def set_general_graphene_nanoribbon_rashba(self, 
            width, length, twist, k, Anderson=-1, λex=0.4, λR=0.3):
    #  _ 
    # / \   Graphene unit cell orientation
    # \_/

    # primitive vectors
    a1 = np.sqrt(3)*np.array([np.sqrt(3)/2, 0.5])
    a2 = np.sqrt(3)*np.array([0,1])
    prims = [a1,  a2]

    orbs = [           "Au",            "Ad",            "Bu",            "Bd"]
    pos  = [np.array([0,0]), np.array([0,0]), np.array([1,0]), np.array([1,0])]
    No = len(orbs)

    # Build the hoppings from the tight-binding functionality
    φ = 0
    t = -1
    hops = []
    hops += bs.graphene(t)
    hops += bs.rashba_phase(λR, φ)
    hops += bs.magnetization(λex)
    
    
    duplicator = bs.hop_utils()

    duplicator.set_prims(prims)
    duplicator.set_orbs(orbs, pos)
    duplicator.set_hops(hops)

    # join unit cell [0,0] with [1,0]
    join = [1,0]

    # New primitive vectors
    A1 = [2,-1]
    A2 = [0, 1]

    duplicator.set_duplication_rules(join, A1, A2)
    duplicator.duplicate_orbs()
    duplicator.duplicate_hops()

    new_A1 = A1[0]*a1 + A1[1]*a2
    new_A2 = A2[0]*a1 + A2[1]*a2
    new_prims = [new_A1, new_A2]

    ham_struct = [duplicator.new_orbs_dic, duplicator.new_pos, new_prims, duplicator.new_hops]
    
    self.set_system(ham_struct, width, length, twist, k)

    C = self.C
    S = self.S

    # if Anderson disorder was not defined
    if type(Anderson) == int:
        Anderson = np.zeros([C,S])
    
    self.Anderson = Anderson
    self.build_spin2() # basis ordering is not the usual. Pauli matrices is also different
    self.build_vels_hsample()


# In[11]:


def set_jaroslav(
    self, width, length, twist, k,t, Δ, λIA, λIB, λR, φ, Anderson=-1):

    #  _ 
    # / \   Graphene unit cell orientation
    # \_/

    # primitive vectors
    a1 = np.sqrt(3)*np.array([np.sqrt(3)/2, 0.5])
    a2 = np.sqrt(3)*np.array([0,1])
    prims = [a1,  a2]

    orbs = [           "Au",            "Ad",            "Bu",            "Bd"]
    pos  = [np.array([0,0]), np.array([0,0]), np.array([1,0]), np.array([1,0])]
    No = len(orbs)

    # Build the hoppings from the tight-binding functionality
    hops = bs.Jaroslav(t,Δ,λIA,λIB,λR,φ)    
    
    duplicator = bs.hop_utils()

    duplicator.set_prims(prims)
    duplicator.set_orbs(orbs, pos)
    duplicator.set_hops(hops)

    # join unit cell [0,0] with [1,0]
    join = [1,0]

    # New primitive vectors
    A1 = [2,-1]
    A2 = [0, 1]

    duplicator.set_duplication_rules(join, A1, A2)
    duplicator.duplicate_orbs()
    duplicator.duplicate_hops()
    # newhops_nocc = bs.remove_cc(duplicator.new_hops)
    newhops_nocc = duplicator.new_hops.copy()

    new_A1 = A1[0]*a1 + A1[1]*a2
    new_A2 = A2[0]*a1 + A2[1]*a2
    new_prims = [new_A1, new_A2]

    ham_struct = [duplicator.new_orbs_dic, duplicator.new_pos, new_prims, newhops_nocc]
    
    

    self.set_system(ham_struct, width, length, twist, k)

    C = self.C
    S = self.S

    # if Anderson disorder was not defined
    if type(Anderson) == int:
        Anderson = np.zeros([C,S])
    
    self.Anderson = Anderson
    self.build_spin2() # basis ordering is not the usual. Pauli matrices is also different
    self.build_vels_hsample()


# In[12]:


def set_branislav(
    self, width, length, twist, k,t, λR, λex, Anderson=-1):

    #  _ 
    # / \   Graphene unit cell orientation
    # \_/

    # primitive vectors
    a1 = np.sqrt(3)*np.array([np.sqrt(3)/2, 0.5])
    a2 = np.sqrt(3)*np.array([0,1])
    prims = [a1,  a2]

    orbs = [           "Au",            "Ad",            "Bu",            "Bd"]
    pos  = [np.array([0,0]), np.array([0,0]), np.array([1,0]), np.array([1,0])]
    No = len(orbs)

    # Build the hoppings from the tight-binding functionality
    
    φ = 0
    hops = []
    hops += bs.graphene(t)
    hops += bs.rashba_phase(λR, φ)
    hops += bs.magnetization(λex)
    
    duplicator = bs.hop_utils()

    duplicator.set_prims(prims)
    duplicator.set_orbs(orbs, pos)
    duplicator.set_hops(hops)

    # join unit cell [0,0] with [1,0]
    join = [1,0]

    # New primitive vectors
    A1 = [2,-1]
    A2 = [0, 1]

    duplicator.set_duplication_rules(join, A1, A2)
    duplicator.duplicate_orbs()
    duplicator.duplicate_hops()

    new_A1 = A1[0]*a1 + A1[1]*a2
    new_A2 = A2[0]*a1 + A2[1]*a2
    new_prims = [new_A1, new_A2]

    ham_struct = [duplicator.new_orbs_dic, duplicator.new_pos, new_prims, duplicator.new_hops]
    
    

    self.set_system(ham_struct, width, length, twist, k)

    C = self.C
    S = self.S

    # if Anderson disorder was not defined
    if type(Anderson) == int:
        Anderson = np.zeros([C,S])
    
    self.Anderson = Anderson
    self.build_spin2() # basis ordering is not the usual. Pauli matrices is also different
    self.build_vels_hsample()


# ## 1D TB

# In[13]:


# just use 2D TB with width=1


# ## 2D TB nanoribbon

# In[14]:


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


# In[15]:


def set_2dtb_nanoribbon(self, width, length, twist, k, ander=0.0):
    # primitive vectors
    a_cc = 1.0
    a1 = np.array([1.0, 0.0])*a_cc
    a2 = np.array([0.0, 1.0])*a_cc
    prim = [a1, a2]

    # orbital positions
    A = np.array([0.0, 0.0])
    
    pos = [A*1.0]

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

# In[16]:


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


# In[17]:


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


# # Green's functions with RGF
# All the Green's functions which are required for the physical quantities

# ## Lead surface Green's functions

# In[18]:


# @jit(nopython=True)
def build_surface_green_right(self,z, niter=60):
    C = self.C
    matE = np.eye(C)*z
    # print(matE)

    a  = self.u*1.0
    b  = self.ut*1.0
    e1 = self.h*1.0
    e2 = self.h*1.0

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
    

    a  = self.ut*1.0
    b  = self.u*1.0
    e1 = self.h*1.0
    e2 = self.h*1.0

    
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

# In[19]:


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

# In[20]:


def build_surfacedrop_GL(self, zs, n, dV):
    C = self.C
    S = self.S

    p = -dV/2
    GL_RGF = self.build_surfaceL(zs+p)*1.0
    
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
    p = -dV/2
    GR_RGF = self.build_surfaceR(zs-p)*1.0
    
    # Only strictly true for graphene nanoribbon - the 0.5 term changes
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


# ## Local Green $G_{nn}$

# In[21]:


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


# ## Local Green $G_{nn}$ drop

# In[22]:


def get_Gnn_drop(self, zs, n, dV):
    
    assert self.ham_defined == True
    C = self.C
    S = self.S
    NZ = len(zs) 

    
    
    # Calculate surface Green's functions next to the slice
    GL_RGF = self.build_surfacedrop_GL(zs,n, dV)
    GR_RGF = self.build_surfacedrop_GR(zs,n, dV)

    # Local Green's function
    length = S*self.prim[0][0] - 0.5
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

# In[23]:


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

# In[24]:


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

# In[25]:


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
    
#     zs_n = zs.conjugate()
    gsurf_RGF_L_p, gsurf_RGF_R_p = self.build_surfaces(zs)
    
    for zz,z in enumerate(zs):
        ΣL_p = VLt@gsurf_RGF_L_p[:,:,zz]@VL
        ΣR_p = VRt@gsurf_RGF_R_p[:,:,zz]@VR
        
        green[:,:,zz] = np.linalg.inv(z*np.eye(C*S)  - self.H_sample - ΣR_p - ΣL_p)


    return green


# ## Local green large drop

# In[26]:


def get_green_large_drop(self, zs, dV):
    # Get the full Green's function inside the sample - 27 Maio 2022
    NE = len(zs)
    

    S = self.S
    C = self.C
    W = self.W
    No = self.No
    # N = S*C

    # Hopping from sample to left lead
    VL = np.zeros([C, C*S], dtype=complex)
    VL[:,:C] = self.u
    VLt = VL.transpose().conjugate()
    
    # Hopping from sample to right lead
    VR = np.zeros([C, C*S], dtype=complex)
    VR[:,C*(S-1):C*S] = self.ut
    VRt = VR.transpose().conjugate()
    
    
    
    # zs_n = zs.conjugate()
    p = -dV/2
    gsurf_RGF_L_p = self.build_surfaceL(zs+p)*1.0
    gsurf_RGF_R_p = self.build_surfaceR(zs-p)*1.0
    
    # Drop  
    xmin = np.min(self.X)
    xmax = np.max(self.X)
    size = xmax - xmin  # extent of the drop
    
    Xp = (self.X - xmin)/size
    drop = dV*(1.0-2*Xp)/2

    # Drop matrix defined in the whole Hilbert space
    dropmat = np.zeros([C*S,C*S], dtype=complex)
    for i in range(S):
        for j in range(W):
            for oo in range(No):
                m = j*No + oo
                n = i*C + m
                dropmat[n,n] = drop[i,m]

    green = np.zeros([C*S,C*S,NE], dtype=complex)
    
    for zz,z in enumerate(zs):
        ΣL_p = VLt@gsurf_RGF_L_p[:,:,zz]@VL
        ΣR_p = VRt@gsurf_RGF_R_p[:,:,zz]@VR
        
        green[:,:,zz] = np.linalg.inv(z*np.eye(C*S) - self.H_sample - dropmat - ΣR_p - ΣL_p)


    return green, dropmat


# # Ozaki integration

# In[27]:


def get_ozaki(N):

    B = np.zeros([N,N])
    for i in range(N-1):
        n = i+1
        b = 0.5/np.sqrt((2*n+1)*(2*n-1))
        B[i,i+1] = b
        B[i+1,i] = b

    vals, vecs = np.linalg.eigh(B)

    # Select the ones that are larger than zero
    assert(N%2==0)
    poles = []
    residues = []
    for i in range(N):
        if vals[i] > 0:
            pole = 1/vals[i]
            res = -abs(vecs[:,i][0])**2/4*pole**2
            poles.append(pole)
            residues.append(res)
    
    return poles, residues


def ozaki_integrator(self, f,N=200):
    # Does not include the first term
   
    # Not repeating the ozaki calculation
    if N not in self.ozaks:
        poles, residues = get_ozaki(N)
        self.ozaks[N] = [poles, residues]
    else:
        poles, residues = self.ozaks[N]
    
    
    terms = []
    for i in range(N//2):

        pole = poles[i]*1j
        res = residues[i]

        # Evaluate the trace at the pole
        term = f(pole)*np.pi*2j*res
        terms.append(term)
        
        # print(term, pole, res)
                
            
    return np.sum(terms)


# # Physical quantities
# Implementation of the formulas that are used to calculate transport properties (Kubo's formula, Keldysh, Landauer)

# ## Landauer

# In[28]:


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

# In[29]:


def kubo_greenwood(self, zs, op1, op2):
    NE = len(zs)    
    zc = zs.conjugate()

    Gp = self.get_green_large(zs) # retarded Green
    Gn = self.get_green_large(zc) # advanced Green
    
    tr = np.zeros(NE, dtype=complex)
    
    for zz,z in enumerate(zs):
        img = (Gp[:,:,zz] - Gn[:,:,zz])/2j        
        tr[zz] = np.trace(img@op1@img@op2)
        
    return tr



# ## Kubo-Bastin

# In[30]:


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

# In[31]:


def kubo_sea_old(self, zs, op, de):
    # returns the integrand of the kubo-Bastin formula which can be used 
    # for several things
    NE = len(zs)
    
    S  = self.S
    C  = self.C
    W  = self.W
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
        img = gp - gn
        
        # tr[zz] = np.trace(img@vx@dg@spinx)/S/S
        tr[zz] = np.trace(img@self.vx@(dgp + dgn)@op)


    return tr#,gz


# In[32]:


def kubo_sea2(self, zs, op, de):
    # returns the integrand of the kubo-Bastin sea formula which can be used 
    # for several things. This uses prebuilt functions
    
    # Imaginary part of zs has to be positive
    zsc = zs.conjugate()
    
    green_R = self.get_green_large(zs)
    green_A = self.get_green_large(zsc)
    
    green_de_R = self.get_green_large(zs  + de)
    green_de_A = self.get_green_large(zsc + de)
     
    NE = len(zs)    
    tr = np.zeros(NE, dtype=complex)
    for zz,z in enumerate(zs):
        
        im = green_R[:,:,zz] - green_A[:,:,zz]
        dgR = (green_de_R[:,:,zz] - green_R[:,:,zz])/de
        dgA = (green_de_A[:,:,zz] - green_A[:,:,zz])/de
        
        dg = dgR + dgA
        tr[zz] = np.trace(self.vx@im@op@dg)

    return tr


# In[33]:


def kubo_sea(self, zs, op, de):
    # returns the integrand of the kubo-Bastin sea formula which can be used 
    # for several things. This uses prebuilt functions
    # This function assumes some symmetries of the Green's functions
    
    green_R = self.get_green_large(zs)
    green_de_R = self.get_green_large(zs  + de)
    
    NE = len(zs)    
    tr = np.zeros(NE, dtype=complex)
    for zz,z in enumerate(zs):
        
        im = green_R[:,:,zz] - green_R[:,:,zz].transpose().conjugate()
        dgR = (green_de_R[:,:,zz] - green_R[:,:,zz])/de
        dgA = (green_de_R[:,:,zz] - green_R[:,:,zz]).transpose().conjugate()/de
        dg = dgR + dgA
        
        tr[zz] = np.trace(self.vx@im@op@dg)

    return tr


# ## Kubo Streda II (analytic sea)

# In[34]:


def kubo_streda_II(self, zs, op, de):
    # returns the Kubo-Streda II integrand term, which can be integrated with Ozaki
    
    green_R = self.get_green_large(zs)
    green_de_R = self.get_green_large(zs  + de)
    
    NE = len(zs)    
    tr = np.zeros(NE, dtype=complex)
    for zz,z in enumerate(zs):
        
        # im = green_R[:,:,zz] - green_R[:,:,zz].transpose().conjugate()
        dgR = (green_de_R[:,:,zz] - green_R[:,:,zz])/de
        # dgA = (green_de_R[:,:,zz] - green_R[:,:,zz]).transpose().conjugate()/de
        # dg = dgR + dgA
        
        tr[zz]  = np.trace(green_R[:,:,zz]@self.vx@dgR@op)
        tr[zz] -= np.trace(dgR@self.vx@green_R[:,:,zz]@op)

    return tr


# In[35]:


def kubozaki_streda(self, mus, op, de, beta = 600, Nozaki=400):
    # Integrate the kubo Sea term with Ozaki
        
    # Function to put inside Ozaki integrator
    def f(mu,x):
        z = x/beta + mu
        zarr = np.array([z])
        tt = self.kubo_streda_II(zarr, op,de)
        
        return tt/beta
    
    NE = len(mus)
    tr = np.zeros(NE, dtype=complex)
    for mm,mu in enumerate(mus):
        #print(f"mu {mu:2.2f}",end="")
        
        def f1(x): return f(mu,x)
        
        soma1 = self.ozaki_integrator(f1,N=Nozaki)
        tr[mm] = soma1

    return tr


# ## Kubo overlap

# In[36]:


def kubo_overlap(self, zs, op):    
    green_R = self.get_green_large(zs)
    
    NE = len(zs)    
    tr = np.zeros(NE, dtype=complex)
    for zz,z in enumerate(zs):
        gr = green_R[:,:,zz]
        ga = gr.transpose().conjugate()
        
        tr[zz]  = np.trace(gr@self.vx@ga@op)
        tr[zz] -= np.trace(ga@self.vx@gr@op)

    return tr


# ## Keldysh

# In[37]:


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


# ## Keldysh sea (with drop)

# In[38]:


def keldysh_sea_drop(self, mus, n, op, dV, eta, beta = 600, Nozaki=400):
    # Integrate the Fermi Sea term with Ozaki
    # W  = self.W
    # No = self.No
    # C  = self.C
    # S  = self.S
    
    
    # Function to put inside Ozaki integrator
    def f(mu,x):
        z = x/beta + mu + 1j*eta
        zarr = np.array([z])
        Gnn = self.get_Gnn_drop(zarr,n, dV)
        green = Gnn[:,:,0]
        tt = np.trace(green@op)
        return tt/beta
    
    
    NE = len(mus)
    tr = np.zeros(NE, dtype=complex)
    for mm,mu in enumerate(mus):
        #print(f"mu {mu:2.2f}",end="")
        
        def fL(x): return f( dV/2 + mu,x)
        def fR(x): return f(-dV/2 + mu,x)
        
        somaL = self.ozaki_integrator(fL,N=Nozaki)
        somaL += -np.pi*1j/2*np.trace(op)

        somaR = self.ozaki_integrator(fR,N=Nozaki)
        somaR += -np.pi*1j/2*np.trace(op)
        
        
        tr[mm] = somaL + somaR

    return tr


# ## Physical quantities wrappers
# Returns the correct normalization constants, and performs extra simulations to check for convergence

# In[39]:


def get_keld_sea(self, energies, n, op_local, dV, eta, beta, Nozaki):

    # Check convergence with dV
    sea2               = self.keldysh_sea_drop(energies, n, op_local, dV*2, eta, beta=beta,   Nozaki=Nozaki)
    sea                = self.keldysh_sea_drop(energies, n, op_local, dV,   eta, beta=beta,   Nozaki=Nozaki)
    sea0               = self.keldysh_sea_drop(energies, n, op_local, dV*0, eta, beta=beta,   Nozaki=Nozaki)
    kelsea             = np.imag(sea -sea0)/dV   + 0*1j
    kelsea2            = np.imag(sea2-sea0)/dV/2 + 0*1j

    # Check convergence with the Ozaki integration
    sea_half           = self.keldysh_sea_drop(energies, n, op_local, dV,   eta, beta=beta,   Nozaki=Nozaki//2)
    sea0_half          = self.keldysh_sea_drop(energies, n, op_local, dV*0, eta, beta=beta,   Nozaki=Nozaki//2)
    kelsea_half        = np.imag(sea_half-sea0_half)/dV + 0*1j
    
    # Check convergence with the finite broadening
    sea_eta_half       = self.keldysh_sea_drop(energies, n, op_local, dV,   eta/2, beta=beta,   Nozaki=Nozaki)
    sea0_eta_half      = self.keldysh_sea_drop(energies, n, op_local, dV*0, eta/2, beta=beta,   Nozaki=Nozaki)
    kelsea_eta_half    = np.imag(sea_eta_half-sea0_eta_half)/dV + 0*1j

    # Check convergence with the temperature
    sea_double_beta    = self.keldysh_sea_drop(energies, n, op_local, dV,   eta, beta=beta*2, Nozaki=Nozaki)
    sea0_double_beta   = self.keldysh_sea_drop(energies, n, op_local, dV*0, eta, beta=beta*2, Nozaki=Nozaki)
    kelsea_double_beta = np.imag(sea_double_beta-sea0_double_beta)/dV + 0*1j
    
    kelsea = np.real(kelsea)
    kelsea2 = np.real(kelsea2)
    kelsea_half = np.real(kelsea_half)
    kelsea_eta_half = np.real(kelsea_eta_half)
    kelsea_double_beta = np.real(kelsea_double_beta)
    #       Normal  dV       Ozaki        eta              beta
    return [kelsea, kelsea2, kelsea_half, kelsea_eta_half, kelsea_double_beta]
    

def get_keld_surface(self, energies, n, op_local, eta):
    zs = energies + 1j*eta
    
    keld1, keld2, keld3 = self.keldysh(zs,n, op_local)
    kelsurf = keld1 - keld2
    
    kelsurf = -0.5*np.real(kelsurf)
    
    return kelsurf

def get_kubo_overlap(self,zs, n, op_local):
    C = self.C
    S = self.S
    a = C*n
    b = a+C
    op_sample = np.zeros([C*S, C*S], dtype=complex)
    op_sample[a:b,a:b] = op_local
    
    surfaced = self.kubo_overlap(zs, op_sample)
    surfaced = -0.5*np.real(surfaced)/self.drop_length
    return surfaced

def get_stredaII(self,zs, n, op_local, de, beta, Nozaki):
    C = self.C
    S = self.S
    a = C*n
    b = a+C
    op_sample = np.zeros([C*S, C*S], dtype=complex)
    op_sample[a:b,a:b] = op_local
    
    ozaked  = self.kubozaki_streda(zs, op_sample, de,   beta,   Nozaki)
    ozaked2 = self.kubozaki_streda(zs, op_sample, de*2, beta,   Nozaki)
    ozaked3 = self.kubozaki_streda(zs, op_sample, de,   beta*2, Nozaki)
    ozaked4 = self.kubozaki_streda(zs, op_sample, de,   beta,   Nozaki//2)
    
    ozaked  = -np.real(ozaked )/self.drop_length
    ozaked2 = -np.real(ozaked2)/self.drop_length
    ozaked3 = -np.real(ozaked3)/self.drop_length
    ozaked4 = -np.real(ozaked4)/self.drop_length
    
    return [ozaked, ozaked2, ozaked3, ozaked4]

def get_kubo_greenwood(self, zs, n, op_local):
    C = self.C
    S = self.S
    a = C*n
    b = a+C
    op_sample = np.zeros([C*S, C*S], dtype=complex)
    op_sample[a:b,a:b] = op_local
    
    greenwood = self.kubo_greenwood(zs, op_sample, self.vx)
    greenwood = -2/self.drop_length*np.real(greenwood)
    
    return greenwood
    


# # Class
# Joins all this functionality into a class

# In[40]:


class rgf:
    ham_defined = False
    h = None
    u = None
    ut = None
    
    # Ozaki poles
    ozaks = {}
    
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

# With drops
rgf.get_Gnn_drop = get_Gnn_drop
rgf.get_green_large_drop = get_green_large_drop

# Physical quantities
rgf.get_landauer = get_landauer
rgf.kubo_bastin = kubo_bastin

rgf.kubo_sea = kubo_sea
rgf.kubo_sea2 = kubo_sea2
rgf.kubo_sea_old = kubo_sea_old
rgf.kubo_streda_II = kubo_streda_II
rgf.kubozaki_streda = kubozaki_streda
rgf.kubo_overlap = kubo_overlap

rgf.kubo_greenwood = kubo_greenwood
rgf.keldysh = keldysh
rgf.keldysh_sea_drop = keldysh_sea_drop
rgf.ozaki_integrator = ozaki_integrator

# Wrappers
rgf.get_kubo_greenwood = get_kubo_greenwood
rgf.get_stredaII       = get_stredaII
rgf.get_kubo_overlap   = get_kubo_overlap
rgf.get_keld_surface   = get_keld_surface
rgf.get_keld_sea       = get_keld_sea

# rgf.set_graphene_nanoribbon_rashba = set_graphene_nanoribbon_rashba
rgf.set_2dtb_nanoribbon = set_2dtb_nanoribbon
rgf.set_2dtb_nanoribbon_large = set_2dtb_nanoribbon_large

rgf.hamiltonian_UC = hamiltonian_UC
rgf.set_system = set_system
rgf.set_general_graphene_nanoribbon_rashba = set_general_graphene_nanoribbon_rashba
rgf.set_jaroslav = set_jaroslav
rgf.set_branislav = set_branislav

rgf.build_spin = build_spin
rgf.build_spin2 = build_spin2
rgf.build_vels_hsample = build_vels_hsample
rgf.generate_hamiltonian = generate_hamiltonian
rgf.hamiltonian_add_drop = hamiltonian_add_drop
rgf.get_eigs = get_eigs

