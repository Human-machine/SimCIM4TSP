import numpy as np 
import torch
import matplotlib.pyplot as plt

def generate_cities(N_cities):
    cities = np.random.randn(N_cities,2)
    L = (cities.reshape(cities.shape[0],1,2) - cities.reshape(1,cities.shape[0],2))**2
    lengths = np.sqrt(np.sum(L,2))
    return cities,lengths

def plot_cities(cities,lengths,order):
    N_cities = cities.shape[0]
    fig, ax = plt.subplots(figsize=(8,5))
    if order.shape[0]==N_cities:
        cities_rearranged = cities[order,:]
        for i in range(N_cities-1):
            ax.plot(cities_rearranged[i:i+2,0],cities_rearranged[i:i+2,1],'-ro')
        ax.plot([cities_rearranged[N_cities-1,0],cities_rearranged[0,0]],
            [cities_rearranged[N_cities-1,1],cities_rearranged[0,1]],'-ro')
        ax.set_xlabel('x',fontsize=18)
        ax.set_ylabel('y',fontsize=18)
        ax.plot(np.zeros(5),'o',color='white',alpha=0.,label =# 'Длина маршрута '+
             str(np.around(length(torch.tensor(order),torch.tensor(lengths)).numpy(),2)))#+'\n алгоритм без нормировки')
        ax.legend(fontsize = 18,loc=1)
        ax.tick_params(labelsize=18)
        ax.grid()
    else:
        ax.text(0.5, 0.5,'Solution is not found',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
    fig.tight_layout()
    return fig

def energy(J,b,s):
    return -0.5*torch.einsum('in,ij,jn->n',s,J,s) - torch.einsum('in,ik->n',s,b)
    
def length(order,lengths):
    order1 = torch.zeros(order.shape[0],dtype=torch.long)
    order1[:-1] = order[1:]
    order1[-1] = order[0]
    return torch.sum(lengths[(order,order1)])   

def Qubo(lengths,A,B):
    N_cities = lengths.shape[0]
    Q = np.zeros((N_cities,N_cities,N_cities,N_cities))
    inds0 = np.arange(N_cities)
    inds1 = np.concatenate((inds0[1:],inds0[0:1]))
    Q[:,inds0,:,inds1] += B*np.repeat(lengths.reshape(1,N_cities,N_cities),N_cities,axis=0)
    dims = np.arange(N_cities)
    Q[dims,:,dims,:] += A
    inds0 = dims.reshape(N_cities,1).repeat(N_cities,axis=1).flatten()
    inds1 = dims.reshape(1,N_cities).repeat(N_cities,axis=0).flatten()
    Q[inds0,inds1,inds0,inds1] -= Q[inds0,inds1,inds0,inds1]
    Q[:,dims,:,dims] += A
    Q[inds1,inds0,inds1,inds0] -= Q[inds1,inds0,inds1,inds0]
    b = -np.ones((N_cities,N_cities))*2*A
    return Q,b

def get_Jh(lengths,A,B):
    N_cities = lengths.shape[0]
    Q,b = Qubo(lengths,A,B)
    Q = torch.tensor(Q.reshape(N_cities**2,N_cities**2),dtype = torch.float32)
    b = torch.tensor(b.reshape(N_cities**2),dtype = torch.float32)
    Q = 0.5*(Q + Q.t())
    J = -0.5*Q
    h = -0.5*(Q.sum(1) + b)
    h = h.reshape(-1,1)
    return J,h

def H(Q,b,x):
    return torch.einsum('ij,ni,nj->n',Q,x,x) + x@b 

def int2base(nums,base,N_digits):
    nums_cur = torch.clone(nums)
    res = torch.empty((nums.shape[0],N_digits))
    for i in range(N_digits):
        res[:,N_digits-1-i] = torch.remainder(nums_cur,base)
        nums_cur = (nums_cur/base).type(torch.long)
    return res

def get_order(x,lengths,A,B):
    Q,b = Qubo(lengths,A,B)
    Q = torch.tensor(Q.reshape(N_cities**2,N_cities**2),dtype = torch.float32)
    b = torch.tensor(b.reshape(N_cities**2),dtype = torch.float32)
    Q = 0.5*(Q + Q.t())
    ind_min = torch.argmin(H(Q,b,x.type(torch.float32)))
    inds_nonzero = np.nonzero(x[ind_min].reshape(N_cities,N_cities))
    inds_order = (inds_nonzero[:,1].sort()[1])
    order = inds_nonzero[:,0][inds_order]
    return order

def get_order_simcim(s_min,N_cities):
    inds_nonzero = np.nonzero((0.5*(s_min+1)).reshape(N_cities,N_cities))
    inds_order = (inds_nonzero[:,1].sort()[1])
    order = inds_nonzero[:,0][inds_order]
    return order