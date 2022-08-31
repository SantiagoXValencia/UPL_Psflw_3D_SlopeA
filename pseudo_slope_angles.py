'''
Pseudoflow algorithm for Ultimate Pit Limit Problem (3D).
Based on work of S Avalos (https://github.com/geometatqueens/2020-Pseudoflow-Python)
The improvement is a variable slope angle for precedences indicated in each block of model
Code not free of bugs
Block model arbitrarily modified for testing, incorporating 4 zones (4 different slope angles)
santiagovalenciam@outlook.com  15/08/2022
'''

import numpy as np
import networkx as NetX
import pseudoflow as pf
import time
import pandas as pd
import math

def precedence(df_model,xsiz,ysiz,zsiz,search_length):
    #List of angles to create a precedence list for each one
    slope_angles=df_model.angulo.unique().tolist()
    #Small block model is created to calculate precedences for each angle
    #   Number of blocks to create
    nbx=int(search_length/xsiz)
    nby=int(search_length/ysiz)
    nbz=int(search_length/zsiz)
    #   Unique values per axis
    xun=np.arange(-nbx*xsiz,(nbx*xsiz)+1,xsiz)
    yun=np.arange(-nby*ysiz,(nby*ysiz)+1,ysiz)
    zun=np.arange(zsiz,(nbz*zsiz)+1,zsiz)
    #Number of items to create
    n_x=len(xun)
    n_y=len(yun)
    n_z=len(zun)
    #Lists of values of x,y and z in small block model
    lista_x=np.repeat(xun,n_y*n_z)
    lista_z=np.tile(zun,n_x*n_y)
    lista_y=np.repeat(yun,n_z)
    lista_y=np.tile(lista_y,n_x)

    #df_preced contains information to export
    df_preced=pd.DataFrame()
    #Creating precedences above central block
    for slopeangle in slope_angles:
        #Creating auxiliar dataframe
        df_aux=pd.DataFrame()
        df_aux['x']=lista_x
        df_aux['y']=lista_y
        df_aux['z']=lista_z
        df_aux['precedencia']=0

        #Looking for precedences inside radius for each level
        df_aux['rad']= (df_aux.z)/math.tan(math.radians(slopeangle))
        df_aux['dist']=(((df_aux.x)**2)+((df_aux.y)**2))**0.5
        df_aux.loc[(df_aux.rad>=df_aux.dist)&(df_aux.rad>0),['precedencia']]=1

        #To reduce procesing time, "repeated blocks" are eliminated
        #repeated blocks are those which are precedences of precedences already considered
        df_aux1=df_aux[(df_aux.precedencia==1)&(df_aux.z==zsiz)]
        for i in df_aux1.index:
            df_aux['prec_aux']=0
            #
            xaux=df_aux1.x[i]
            yaux=df_aux1.y[i]
            zaux=df_aux1.z[i]
            #
            df_aux['rad']= (df_aux.z-zaux)/math.tan(math.radians(slopeangle))
            df_aux['dist']=(((df_aux.x-xaux)**2)+((df_aux.y-yaux)**2))**0.5
            df_aux.loc[(df_aux.rad>=df_aux.dist)&(df_aux.rad>0),['prec_aux']]=1 
            #
            df_aux.loc[(df_aux.precedencia==1)&(df_aux.prec_aux==1),['precedencia']]=0
        #
        df_aux['angulo']=slopeangle
        df_aux=df_aux[df_aux.precedencia==1]
        df_preced=df_preced.append(df_aux, ignore_index=True)
    ####
    df_preced=df_preced[['x','y','z','angulo']]
    df_preced['x']=df_preced['x']/xsiz
    df_preced['y']=df_preced['y']/ysiz
    df_preced['z']=df_preced['z']/zsiz
    np_preced=np.array(df_preced) #numpy format reduces time
    return (np_preced)

def reduce_model(BM, zsiz, xmn, ymn, zmn):
    max_slope=max(BM[:,6])
    delta=zsiz/(math.tan(math.radians(max_slope)))
    xmx= max(BM[:,0])
    ymx= max(BM[:,1])
    zmx= max(BM[:,2])
    lblocks=[]
    levels=np.arange(zmx-zsiz,zmn-1,-zsiz)
    #Adding all blocks of highest level
    laux=np.where(BM[:,2]==zmx)
    lblocks.extend(laux[0].tolist())
    #Other levels
    c=1
    for i in levels:
        laux=np.where((BM[:,2]==i)&(BM[:,0]>=(xmn+(delta*c)))&(BM[:,0]<=(xmx-(delta*c)))&(BM[:,1]>=(ymn+(delta*c)))&(BM[:,1]<=(ymx-(delta*c))))
        lblocks.extend(laux[0].tolist())
        c+=1
    lblocks=set(lblocks) #Working with set reduces time
    return(lblocks)


def Pseudoflow_UPL(lblocks, precedence_matrix, xmn, ymn, zmn, xsiz, ysiz, zsiz, BM, nx, ny, nz, VarIn, VarOut):
    print("Beginning Pseudoflow")
    start_UPL = time.time() 
    source = 0
    sink = np.int(nx*ny*nz + 1)
    
    # Graph creation
    Graph = NetX.DiGraph()

    
    # External arcs creation by external function. Source - Nodes, Nodes - Sink
    Graph = CreateExternalArcs(lblocks, BM, nx, ny, nz, Graph=Graph, Var=VarIn)
    
    # Internal arcs creation by external function. Block precedence
    for i in lblocks:
        pos_x=int((BM[i][0]-xmn)/xsiz)
        pos_y=int((BM[i][1]-ymn)/ysiz)
        pos_z=int((BM[i][2]-zmn)/zsiz)
        Graph = CreateInternalArcs(lblocks, precedence_matrix, BM, pos_x, pos_y, pos_z, nx, ny, nz, Graph=Graph)
    
    # Solving the minimum cut problem via pf.hpf solver
    RangeLambda = [0]
    breakpoints, cuts, info = pf.hpf(Graph, source, sink, const_cap="const", mult_cap="mult", lambdaRange=RangeLambda, roundNegativeCapacity=False)
    
    #Going over the cuts.items finding the nodes inside the resulting UPL.
    B = {x:y for x, y in cuts.items() if y == [1] and x!=0}
    InsideList = list(B.keys())
    
    # Set all blocks as zero
    BM[:,VarOut] = 0 

    for indUPL in range(len(InsideList)): 
        # Set blocks inside UPL as one
        BM[np.int(InsideList[indUPL] -1),VarOut] = 1
    
    print("--> Pseudoflow time: --%s seconds " % (np.around((time.time() - start_UPL), decimals=2)))  

    return BM

def CreateExternalArcs(lblocks, BM, nx, ny, nz, Graph, Var):
    Sink = np.int(nx*ny*nz + 1)

    for p_i in lblocks:
        Capacity = np.absolute(np.around(BM[p_i,Var], decimals=2))
        if BM[p_i,Var] < 0: #Negative local Economic Value
            Graph.add_edge(p_i+1, Sink, const=Capacity, mult=-1)
        else:
            Graph.add_edge(0, p_i+1, const=Capacity, mult=1)
    return Graph


def CreateInternalArcs(lblocks, precedence_matrix, BM, pos_x, pos_y, pos_z, nx, ny, nz, Graph):
    p_0 =  pos_x + nx*pos_y + ny*nx*pos_z #Index of block analyzed
    prece_por_buscar=np.where(precedence_matrix[:,3]==BM[p_0][6])

    for i in prece_por_buscar[0]: #For each relative precedence
        bpx=pos_x+precedence_matrix[i][0]
        bpy=pos_y+precedence_matrix[i][1]
        bpz=pos_z+precedence_matrix[i][2]
        if (bpx>=0)and(bpx<nx)and(bpy>=0)and(bpy<ny)and(bpz<nz):
            bloque_preced= bpx + nx*bpy + ny*nx*bpz
            if bloque_preced in lblocks:
                Graph.add_edge(p_0+1, bloque_preced+1, const=99e99, mult=1)

    return Graph

def main():    
    print("Start")
    start_time = time.time() 
    nx, xmn, xsiz = 44, 24300, 16
    ny, ymn, ysiz = 62, 24800, 16
    nz, zmn, zsiz = 26, 3600, 16
    search_length=32 #To calculate block relative precedences

    df_model=pd.read_csv('BM_matrix.txt') # Import Block Model 
    precedence_matrix=precedence(df_model,xsiz,ysiz,zsiz,search_length)
    print("--%s seconds calculating precedence matrix-" % (np.around((time.time() - start_time), decimals=2)))

    BlockModel = np.array(df_model) #BM in numpy format

    #Reducing block model to max slope angle
    lblocks=reduce_model(BlockModel, zsiz, xmn, ymn, zmn)

    #Running pseudoflow
    BlockModel = Pseudoflow_UPL(lblocks, precedence_matrix, xmn, ymn, zmn, xsiz, ysiz, zsiz, BM=BlockModel, nx=nx, ny=ny, nz=nz, VarIn=4, VarOut=5)
    
    '''Save Block Model'''
    # np.savetxt(fname="BM_matrix.txt", X=BlockModel, fmt='%.3f', delimiter='\t')
    df_model=pd.DataFrame(data=BlockModel,columns=df_model.columns)
    df_model.to_csv('BM_matrix.txt',index=False)
          
    return print("--%s seconds of the whole process-" % (np.around((time.time() - start_time), decimals=2)))  


if __name__ == "__main__":
    main()