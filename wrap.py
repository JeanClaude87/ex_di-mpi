from mpi4py import MPI
import numpy as np
import f_diagonal as diagonal
import f_function as ff
import os

LOCAL = os.path.abspath('.')

#....................................DISORDINE
D_i = 1.0
D_f = 10.0
D_D = 1.0

D_n = int(1+(D_f-D_i)/D_D)

#print D_i, D_f, D_n
D_tab = [D_i+j*D_D for j in range(D_n)]


#....................................LUNGHEZZA
L_i = 8
L_f = 10
L_D = 2

L_n = int(1+(L_f-L_i)/L_D)

#print L_i, L_f, L_n
L_tab = [L_i+j*L_D for j in range(L_n)]

for L in L_tab:

	nomefile = 'CdC-L_'+str(L)+'.npy'
	if not os.path.isfile(LOCAL+os.sep+nomefile):
		CdC_Tab = ff.prep_tab(L)
		np.save(LOCAL+os.sep+nomefile, CdC_Tab)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



if rank == 0:
	for i in L_tab:
       		for j in D_tab:
               		directory = 'L_'+str(i)+'/D_'+str(j)
               		if not os.path.exists(LOCAL+os.sep+directory):
                       		os.makedirs(LOCAL+os.sep+directory)


#....................................NN Realiz
NN_RR = [50] #*30
N_proc = size

n0=0
for i in L_tab:

	nomefile = str('CdC-L_'+str(i)+'.npy')
	Tab_CdC = np.load(LOCAL+os.sep+nomefile)

	for j in D_tab:
		directory = 'L_'+str(i)+'/D_'+str(j)
		PATH_now = LOCAL+os.sep+directory+os.sep
		
		for n in range(NN_RR[n0]):
			data = [i,j,n+1]

			diagonal.ExactDiagonalization(PATH_now,data[0],data[1],Tab_CdC)
			#def 	 ExactDiagonalization(PATH_now,   L,     D,    Tab_CdC):			
	n0 += 1







