#Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

#Other imports
import numpy as np
import matplotlib.pyplot as plt
import torch

datatype = torch.float32
device = 'cuda'

class Simcim(mli.Interface):
    
    params_disc = {}
    params_cont = {}

    params_cont['c_th'] = 1.
    params_cont['zeta'] = 1.

    params_disc['N'] = 2000
    params_disc['attempt_num'] = 200
    params_cont['dt'] = .1
    params_cont['sigma'] = .01
    params_cont['alpha'] = 0.9

    params_cont['S'] = torch.tensor(0.1)
    params_cont['D'] = torch.tensor(-0.8)
    params_cont['O'] = torch.tensor(0.05)    
        
    def __init__(self,J,b,device,datatype):
        super(Simcim,self).__init__()

        self.J = J.type(datatype).to(device)
        self.b = b.type(datatype).to(device)
        self.Jmax = torch.max(torch.sum(torch.abs(J),1))
        self.dim = J.shape[0]
        
        self.device = device
        self.datatype = datatype
        
    
    # amplitude increment
    def ampl_inc(self,c,p):
        return ((p*c + self.zeta*(torch.mm(self.J,c)+self.b))*self.dt
            + (self.sigma*torch.randn((c.size(0),self.attempt_num),dtype=self.datatype).to(self.device)))
    
    # pump ramp
    def pump(self):
        i = torch.arange(self.N,dtype=self.datatype).to(self.device)
        arg = torch.tensor(self.S,dtype = self.datatype).to(self.device)*(i/self.N-0.5)
        return self.Jmax*self.O*(torch.tanh(arg) + self.D )

    def pump_lin(self):
        t = self.dt*torch.arange(self.N,dtype=self.datatype).to(self.device)
        eigs = torch.eig(self.J)[0][:,0]
        eig_min = torch.min(eigs)
        eig_max = torch.max(eigs)
        p = -self.zeta*eig_max + self.zeta*(eig_max-eig_min)/t[-1]*t
        return p
    
    # amplitude initializer 
    def init_ampl(self):
        return torch.zeros((self.dim, self.attempt_num),dtype=self.datatype).to(self.device)
    
    def tanh(self,c):
        return self.c_th*torch.tanh(c)
    
    # evolution of amplitudes
    # N -- number of time iterations
    # attempt_num -- number of runs 
    # J -- coupling matrix
    # b -- biases
    # O, S, D -- pump parameters
    # sigma -- sqrt(std) for random number
    # alpha -- momentum parameter
    # c_th -- restriction on amplitudes growth
    def evolve(self,params_opt):
        
        for key in params_opt.keys():
            self.params_cont[key] = params_opt[key]

        self.N = self.params_disc['N']
        self.attempt_num = self.params_disc['attempt_num']
        self.dt = self.params_cont['dt']
        self.zeta = self.params_cont['zeta']
        self.c_th = self.params_cont['c_th']
        self.O = self.params_cont['O']
        self.D = self.params_cont['D']
        self.S = self.params_cont['S']
        self.sigma = self.params_cont['sigma']
        self.alpha = self.params_cont['alpha']
        self.Jmax = torch.max(torch.sum(torch.abs(self.J),1))
        self.dim = self.J.shape[0]
        
    
        # choosing an attempt for the amplitude evolution 
        random_attempt = np.random.randint(self.attempt_num)
    
        # initializing current amplitudes
        c_current = self.init_ampl()
    
        # initializing full array of amplitudes
    #     c_full = torch.zeros(N,dim,attempt_num)
    #     c_full[0] = c_current
    
        # creating the array for evolving amplitudes from random attempt
        c_evol = torch.empty((self.dim, self.N),dtype=self.datatype).to(self.device)
        c_evol[:,0] = c_current[:,random_attempt]
    
        # define pump array
        p = self.pump()
#         p = self.pump_lin()
    
        # define copupling growth
    #     zeta = coupling(init_value,final_value,dt,N)
    
        # initializing moving average of amplitudes increment
        dc_momentum = torch.zeros((self.dim, self.attempt_num),dtype=self.datatype).to(self.device)
        #free_energy_ar = torch.empty(self.N-1, dtype = self.datatype).to(device)
        for i in range(1,self.N):
        
            # calculating amplitude increment
            dc = self.ampl_inc(c_current,p[i])
            dc /= torch.sqrt((dc**2).sum(0)).reshape(1,self.attempt_num)
        
            # calculating moving average of amplitudes increment
            dc_momentum = self.alpha*dc_momentum + (1-self.alpha)*dc
        
            # calculating possible values of amplitudes on the next step
            c1 = c_current + dc_momentum
        
            # comparing c1 with c_th
            th_test = (torch.abs(c1)<self.c_th).type(self.datatype)
        
            # updating c_current
    #         c_current = c_current + th_test*dc_momentum
            c_current = th_test*(c_current + dc_momentum) + (1.-th_test)*torch.sign(c_current)*self.c_th
    #         c_current = step(c_current + dc_momentum,c_th,device, datatype)
    #         c_current = tanh(c1,c_th)
        
            #updating c_full
    #         c_full[i] = torch.tanh(c_full[i-1] + dc_momentum)
        
            # add amplitude values from random attempt to c_evol array 
            c_evol[:,i] = c_current[:,random_attempt]
            
            #s = torch.sign(c_current)
            #free_energy_ar[i-1] = self.free_energy(s,self.beta)
        
        return c_current, c_evol
    
    def energy_cost(self,params_opt):
        c_current, c_evol = self.evolve(params_opt)
        s = torch.sign(c_current)
        return -0.5*torch.einsum('in,ij,jn->n',s,self.J,s) - torch.einsum('in,ik->n',s,self.b)
    
    def energy(self,s):
        return -0.5*torch.einsum('in,ij,jn->n',s,self.J,s) - torch.einsum('in,ik->n',s,self.b)
    
    
    def get_next_cost_dict(self,params_dict):        
        #Get parameters from the provided dictionary
        params = params_dict['params']
        params_opt = {}
        
        params_opt['O'] = params[0]
        params_opt['S'] = params[1]
        params_opt['D'] = params[2]
        params_opt['zeta'] = params[3]
        
        
        cost = torch.min(self.energy_cost(params_opt)).cpu().numpy()
        #There is no uncertainty in our result
        uncer = 0
        #The evaluation will always be a success
        bad = False
        #Add a small time delay to mimic a real experiment
        
        #The cost, uncertainty and bad boolean must all be returned as a dictionary
        #You can include other variables you want to record as well if you want
        cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
        return cost_dict
    
def main(controller_type,interface):
    
    controller = mlc.create_controller(interface,
                                       controller_type = controller_type,
                                       max_num_runs_without_better_params = 30,
                                       max_num_runs = 100,
                                       num_params = 4, 
                                       min_boundary = [0,0,-2,0],
                                       max_boundary = [3,3,2,10])
    
    controller.optimize()
    
    
    print('Best parameters found:')
    print(controller.best_params)
    
#     mlv.show_all_default_visualizations(controller)
    params = controller.best_params
    params_opt = {} 
    params_opt['O'] = params[0]
    params_opt['S'] = params[1]
    params_opt['D'] = params[2]
    params_opt['zeta'] = params[3]
    return params_opt
    