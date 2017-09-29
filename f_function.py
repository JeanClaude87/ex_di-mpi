import numpy as np
import os.path
import math
import scipy.linalg as _la
from math import factorial
import itertools
import time 
import scipy.special as special
import os
from datetime import datetime
import time


#..................................counting number of zero
POPCOUNT_TABLE16 = [0] * 2**16
for index in xrange(len(POPCOUNT_TABLE16)):
	POPCOUNT_TABLE16[index] = (index & 1) + POPCOUNT_TABLE16[index >> 1]

def one_count(v):
	return (POPCOUNT_TABLE16[ v        & 0xffff] +
			POPCOUNT_TABLE16[(v >> 16) & 0xffff])


#..................................Binomial
def comb(n, k):
	kk = factorial(n) / factorial(k) / factorial(n - k)
	uga= int(kk)
	return uga


#..................................from configuration to bin number
def TO_bin(xx):
	return int(xx,2)

#..................................from bin number to configuration
def TO_con(x,L):
	return np.binary_repr(x, width=L)


#..................................base preparation
def Base_prep(n,k):
	result = []
	for bits in itertools.combinations(range(n), k):
		s = ['0'] * n
		for bit in bits:
			s[bit] = '1'
		result.append(''.join(s))
	return result

def BaseNumRes_creation(Dim,LL,B):
	A=np.zeros((Dim,LL), dtype=np.float)

	for i in range(Dim):
		k=0
		for j in list(B[i]):
			A[i,k] = float(j)-0.5
			k+=1
	return A

#..................................hop. preparation
def Hop_prep(L,BC):
	if BC == 1:
		Hop_dim=L-1
	else:
		Hop_dim=L
	return [TO_con(2**i+2**((i+1)%L),L) for i in range(Hop_dim)]


#..................................................Disorder creation
def Dis_Creation(LL,Dis_gen):

	dis = np.zeros(LL, dtype=np.float)
	for i in range(LL):
		if Dis_gen==0: 
			dis[i] = 2*np.random.random()-1
		else:
			dis[i] = np.cos(2*math.pi*0.721*i/LL)
	return dis


#..................................creation Lin Tables
def LinTab_Creation(L,Base,Dim):

#..........................Table Creation
	MaxSizeLINVEC = sum([2**(i-1) for i in range(1,L/2+1)])

	#....creates a table LinTab_L+LinTab_R
	#.....................[  ,  ]+[  ,  ]
	LinTab = np.zeros((MaxSizeLINVEC+1,4),dtype=int)
	Jold=JJ=j1=j2=0
	Conf_old= TO_con(0,L/2)

#...........................Table Filling
	for i in range(Dim):
		Conf_lx = Base[i][0:L/2]
		Bin_lx  = TO_bin(Conf_lx)
		Conf_rx = Base[i][L/2:L]		
		Bin_rx  = TO_bin(Conf_rx)

		if Conf_lx==Conf_old:
			j1 = Jold
		else:
			j1 += j2

		Conf_old = Conf_lx 

		if Jold != j1:	
			JJ = Jold = 0

		j2	= JJ+1
		Jold = j1
		JJ  += 1

		#print Conf_lx, int(Bin_lx), int(j1), Conf_rx, int(Bin_rx), int(j2)

		LinTab[Bin_lx,0]= int(Bin_lx)
		LinTab[Bin_lx,1]= int(j1)
		LinTab[Bin_rx,2]= int(Bin_rx)
		LinTab[Bin_rx,3]= int(j2)

#	print LinTab
	return LinTab

#..................................Lin Look for complete state
def LinLook(vec,LL,arr):

	Vec  = TO_con(vec,LL)
	v1	 = Vec[0:LL/2]
	v2	 = Vec[LL/2:LL]
	ind1 = TO_bin(v1)
	ind2 = TO_bin(v2)
	return arr[ind1,1]+arr[ind2,3]-1

#..................................Lin Look for RIGHT state
def LinLook_LL(vec,arr):
	ind=TO_bin(vec)
	return arr[ind+1,1]


#..................................Lin Look for RIGHT state
def LinLook_RR(vec,arr):
	ind=TO_bin(vec)
	return arr[ind+1,3]


#..................................................Hamiltonian Creation
def Ham_Dense_Creation(LL,NN,Dim,D,Dis_real,BC,Base_Bin,Base_Num,Hop_Bin,LinTab):

	t=1.
	# tutto in unita di t!!

	ham = np.zeros((Dim,Dim), dtype=np.float)

	if BC == 1:
		Hop_dim=LL-1
	else:
		Hop_dim=LL

	for i in range(Dim):
		n_int = 0.0
		n_dis = 0.0
		bra = LinLook(Base_Bin[i],LL,LinTab)

		for j in range(Hop_dim):
			xx  = Base_Bin[i]^Hop_Bin[j]
			ket = LinLook(xx,LL,LinTab)
			
			if one_count(xx) == NN:
				ham[bra,ket] = t/2
				#ham[bra,ket] = t 
			uu = Base_Bin[i] & Hop_Bin[j]
			
			if one_count(uu) == 1:
				n_int -= 0.25
				#0.5 perche spin 1/2*1/2
			else: 
				n_int += 0.25
			
			#print TO_con(Base_Bin[i],LL), TO_con(Hop_Bin[j],LL), TO_con(uu,LL), one_count(uu), n_int

			n_ones = Base_Bin[i] & int(2**(LL-j-1)) 
			#diventa diverso da zero solamente se ce un 1 in quel sito
			if n_ones != 0:
				n_dis += 0.5*Dis_real[j]
				#0.5 perche spin 1/2
			else:
				n_dis -= 0.5*Dis_real[j]

		ham[bra,bra] = t*(n_int + D*n_dis)
		#print TO_con(bra,LL), n_int
	return ham


#..................................................Hamiltonian Dense Diagonalization
def eigval(A):
	E = _la.eigh(A)
	return E

#..................................................Hamiltonian Dense Diagonalization
def levstat(E,Dim):
	gap=E[1:]-E[:-1]
	B = np.zeros(Dim-2, dtype=np.float)
	for i in range(Dim-2):
		B[i]=np.minimum(gap[i+1],gap[i])/np.maximum(gap[i+1],gap[i])	
	return B

#..................................................Hamiltonian Sparse Diagonalization
def eigsh(A,n):
	E = _la.sparse.linalg.eigsh(A, n=6)
	return E

#..................................................Initial state
def Psi_0(Dim):
	n = np.random.randint(0,Dim-1)
	return n

def Proj_Psi0(a,V):
	return V[a]**2


#..................................................Traslations MEAN
def Trasl_Mean(A):
	a = A.shape
	B = np.zeros((a[1],a[1]), dtype=np.float)
	for i in range(a[1]):
		B[i] = np.roll(A[i],-i)
	return np.mean(B, axis=0)



#..................................................NiNj
def OUTER_creation(L,Dim,A):
	B = np.zeros((Dim,L,L), dtype=np.float)
	for i in range(Dim):
		B[i] = np.outer(A[i],A[i])
	return B

def Mat_Corr_i(A,B):
	corr_zero=np.einsum('il,ijk -> ljk', A**2, B)
	return corr_zero

def Mat_Corr_Psi0(A,B,C):
	corr_zero=np.einsum('l,il,ijk -> jk',C, A**2, B)
	return corr_zero

def Mat_CorrConn_Psi0(A,B,C,D):
	corr_zero=np.einsum('il,ijk -> ljk', A**2, B)	
	corr_zero_conn = corr_zero-D
	corr_conn = np.einsum('i,ijk -> jk', C, corr_zero_conn)

	return corr_conn

def Mat_Corr_MiCa(A,B):
	corr_zero=np.einsum('il,ijk -> jk', B, A**2)
	return corr_zero


#..................................................CdiCj

def prep_tab(L):
	Dim = comb(L, L/2)

	Base_Num = Base_prep(L,L/2)
	Base_Bin = [int(Base_Num [i],2) for i in range(Dim)]
	LinTab   = LinTab_Creation(L,Base_Num,Dim)

	CdC_Tab  = CdC_tabCreation (L,L/2,Dim,Base_Num,Base_Bin,LinTab)

	return CdC_Tab

def CdC_tabCreation (LL,NN,Dim,Base_Num,Base_Bin,LinTab):
	dimCiCj =  comb(LL-2, NN-1)
	CdC_Tab  =  np.zeros((LL,LL,dimCiCj,2), dtype=int)
	
	for i in range(LL):
		for j in range(LL):

			xx = np.zeros((dimCiCj,2), dtype=int)
			x0 = 0

			for l in range(Dim):

				a = Base_Num[l][0:i] 
				b = Base_Num[l][i+1:LL]
				c = ''.join([a,'1',b])
				a = c[0:j] 
				b = c[j+1:LL]
				d = ''.join([a,'0',b])
				if (one_count(int(d,2)) == NN and int(d,2) != Base_Bin[l]):
					bra = LinLook(Base_Bin[l],LL,LinTab)
					ket = LinLook(int(d,2),LL,LinTab)
				
					xx[x0,0] = int(bra)
					xx[x0,1] = int(ket)
					x0 += 1
				
			CdC_Tab[i,j] = xx

	return CdC_Tab

def Mat_CdC_i(UU1,LL,V,l):

	CC = np.zeros((LL,LL),dtype=float)

	for i in range(LL):
		for j in range(i,LL):
			uu = UU1[i,j]
			CC[j,i] = CC[i,j] = np.inner(V[uu[:,0],l],V[uu[:,1],l])
	np.fill_diagonal(CC, 0.25)
	return CC

def Mat_CdC_Psi0(UU1,Proj_Psi0,Dim,LL,V):

	CC = np.empty((Dim,LL,LL),dtype=float)

	for l in range(Dim):
		for i in range(LL):
			for j in range(i,LL):
				uu = UU1[i,j]
				CC[l,j,i] = CC[l,i,j] = np.inner(V[uu[:,0],l],V[uu[:,1],l])
				#print Dim, l, i, j, np.inner(V[uu[:,0],l],V[uu[:,1],l])
		np.fill_diagonal(CC[l], 0.25)
		
		CC[l] *= Proj_Psi0[l]

	CC1 = np.einsum('ijk -> jk', CC)

	return CC1

def generate_filename(basename):
	unix_timestamp = int(time.time())
	local_time = str(int(round(time.time() * 1000)))
	xx = basename + local_time + ".dat"
	if os.path.isfile(xx):
		time.sleep(1)
		return generate_filename(basename)
	return xx



#..................................................Entropy

#..................................................Time - Evolution

def Proj_t(t,Proj_Psi0,E):
	xx = Proj_Psi0*np.exp(-1j*E*t)
	return xx







