import numpy as np
import time 
import scipy.special as special
import f_function as ff

def ExactDiagonalization(PATH_now,L,D,Tab_CdC):

	#here LL is the number L is the string
	# EVERYTHING IS IN UNIT OF t
	# t ---> is set to 1


	t0=time.time()
	#.PARAMETERS.......................................boundary conditions
	#...................BC=0 periodic, BC=1 open
	BC		= 0				
	#..................................................disorder parameters
	#...................dis_gen=0 random, dis_gen=1 quasiperiodic
	Dis_gen = 0

	#..................................................Supspace dimension
	LL = int(float(L))
	DD = float(D)

	NN  = int(LL/2)
	Dim = ff.comb(LL, NN)

	#..................................................Base creation
	Base_Num = ff.Base_prep(LL,NN)
	Base_Bin = [int(Base_Num [i],2) for i in range(Dim)]

	#..................................................Hopping creation
	if BC == 1:
		Hop_dim=LL-1
	else:
		Hop_dim=LL

	Hop_Num = ff.Hop_prep(LL,BC)
	Hop_Bin = [int(Hop_Num[i],2) for i in range(Hop_dim)]

	#..................................................Lin Tab creation
	LinTab = ff.LinTab_Creation(LL,Base_Num,Dim)

	#.............................Disorder creation
	Dis_real = ff.Dis_Creation(LL,Dis_gen)

	#.............................Diagonalization HAM
	HAM   = ff.Ham_Dense_Creation(LL,NN,Dim,DD,Dis_real,BC,Base_Bin,Base_Num,Hop_Bin,LinTab)
	#print HAM

	E,V   = ff.eigval(HAM)	

	#print E


	levst = ff.levstat(E,Dim)
	
	m_levst = np.mean(levst)
	print m_levst
	
	nomefile_lev = str(PATH_now+'levst.dat')
	with open(nomefile_lev, 'a') as ee:
		ee.write('%f' % m_levst+"\n")



	'''
	#V[Psi0] proiezioni
	#V[:,Psi0] autovettori

	#.............................Initial state
	Psi0		= ff.Psi_0(Dim)
	Proj_Psi0   = ff.Proj_Psi0(Psi0,V)


	entropy = -np.sum(Proj_Psi0*np.log2(Proj_Psi0))

	nomefile_ent = str(PATH_now+'entr-')
	with open(ff.generate_filename(nomefile_ent), 'w') as ee:
		ee.write('%f' % entropy)

	#.............................Densita
	Base_NumRes = ff.BaseNumRes_creation(Dim,LL,Base_Num)
	Base_Corr	= ff.OUTER_creation(LL,Dim,Base_NumRes)

	Dens		= np.dot(np.transpose(V**2),Base_NumRes)
	DensDens	= ff.OUTER_creation(LL,Dim,Dens)

	#.............................NN
	NN			= ff.Mat_Corr_Psi0(V,Base_Corr,Proj_Psi0)
	NN_Conn	= ff.Mat_CorrConn_Psi0(V,Base_Corr,Proj_Psi0,DensDens)
	
	#NN_Conn_tr = ff.Trasl_Mean(NN_Conn)

	nomefile_NN = str('corr_n-')
	np.savetxt(ff.generate_filename(PATH_now+nomefile_NN), NN_Conn, fmt='%.9f')



	#.............................CiCj
	CdC    = ff.Mat_CdC_Psi0(Tab_CdC,Proj_Psi0,Dim,LL,V)
	#	CdC_tr = ff.Trasl_Mean(CdC)

	nomefile_cc = str('corr_c-')
	np.savetxt(ff.generate_filename(PATH_now+nomefile_cc), CdC, fmt='%.9f')
	'''
	return 1







