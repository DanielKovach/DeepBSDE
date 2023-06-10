import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time

torch.manual_seed(1234) 

#solution to PDE
def solution_u(x,t):
    return np.exp(-3*t)*(np.square(x))

#true gradient of solution
def grad_u(x,t):
    return np.exp(-3*t)*2*x

#terminal condition
def g(x,T):
    return np.exp(-3*T)*(np.square(x))

#gamma(x) = x^2
def gamma(x):
    return x**2

#sigma = Identity matrix
def sigma(X):
    return X

#mu(x) = x
def mu(X):
    return X

#nonlinear function in general, in our case 0
def f(t,X,U,S):
    return 0

#Approximate Brownian motion
def BM_approx(N,M,T):
    
    dt = T/(N-1)
    dx = np.sqrt(dt)
    BM = torch.zeros((N, M))

    steps = 1-2*torch.bernoulli(torch.empty(N, M).uniform_(0, 1)) #random -1 or 1 for matrix of size N x M-1

    for n in range(1,N):
            BM[n,:] = torch.add(BM[n-1,:], dx*(steps[n-1,:]))        
  
    return BM
 
# Takes in a choice of gamma/sigma/mu and produces the X_t paths at times
#t_n corresponding to a discrete set of omega's 
def X_path(N,M,T,xi):

    dt = T/(N-1)
    X = torch.zeros((N, M))
    X[0,:] = xi
    BM = BM_approx(N,M,T)
    dW = torch.zeros((N, M))
    for n in range(1,N):
            dW[n,:] = torch.add(BM[n,:],-BM[n-1,:])
            X[n,:] = torch.add(X[n-1,:], torch.add(dt*mu(X[n-1,:]), torch.mul(sigma(X[n-1,:]),dW[n,:])))
  
    return dW, X

#Equation 5 from [HJE18], only used for plotting
def pde_solution_true(N,M,T,xi):
    
   dt = T/(N-1)
   U = torch.zeros((N, M))
   U[0,:] = solution_u(xi,0) #cheating
   X_p = X_path(N,M,T,xi)
   dW = X_p[0]
   X = X_p[1]        

   for n in range(1,N):
           #U[n,:] = torch.add(U[n-1,:] ,torch.add(-dt*f(dt*n,X[n-1,:],U[n-1,:],sigma(grad_u(X[n-1,:],dt*n))), torch.mul(sigma(grad_u(X[n-1,:],dt*n)),(dW[n,:]))))
           U[n,:] = torch.add(U[n-1,:], torch.mul(sigma(grad_u(X[n-1,:],dt*n)),(dW[n,:])))       
            
   return U

#Equation 5 from [HJE18]
def pde_solution_approx(N,M,T,xi,dW,X,G_U,u_0_xi_hat):
   
   dt = T/(N-1)
   U = torch.zeros((N, M))
   U[0,:] = u_0_xi_hat 

   X_p = X_path(N,M,T,xi)
   dW = X_p[0]
   X = X_p[1]        

   for n in range(1,N):
           #U[n,m] = torch.add(U[n-1,m], torch.add(-dt*f(dt*n,X[n-1,m],U[n-1,m],G_U[n-1,m]), torch.mul(G_U[n-1,:],(dW[n,:]))))
           U[n,:] = torch.add(U[n-1,:], torch.mul(G_U[n-1,:],(dW[n,:])))
  
   return U

def back_build_pde_solution_approx(N,M,T,xi,dW, X, GU,u_0_xi_hat):

   dt = T/(N-1)
   U = torch.zeros((N, M))
   U[N-1,:]=g(xi,T) + u_0_xi_hat #g(xi) + u(0,xi)
   for n in range(N-1,0,-1): #iterate backwards from N-1 to 1
           #U_n_minus_approx = torch.add(U[n,:],-dt*grad_u(X[n,:],dt*n)) #Approximate U[n-1,:] input needed for f with U[n,:]-grad_u(X[n,:],dt*n)*dt.
           #U[n-1,:] = torch.add(U[n,:], torch.add(dt*f(dt*n,X[n-1,:],U_n_minus_approx,GU[N-1,:]), torch.mul(-GU[N-1,:],dW[n,:]))) 
           U[n-1,:] = torch.add(U[n,:] ,torch.mul(-GU[n-1,:],dW[n,:]))            #if f = 0
  
   return U

#compute u(0,xi) through Y_0
def compute_u_0_X0(N,M,T,dW, X, GU):
     U = torch.zeros((N, M))
     U[0,:]=g(X[N-1,:],T)
     for n in range(1,N):
          U[n,:] = torch.add(torch.mul(-GU[n-1,:],dW[n,:]),U[n-1,:])
     return torch.mean(U[N-1,:]) 

######################################################################################
#Main
######################################################################################

###########################################
#Parameters
###########################################

N = 30 
M = 1000
T = 1
xi = 1

layer_size = 5 #layer_size = 4 in [HJE18]
learning_rate = 0.01 #same as [HJE18]
#iteration_num = 40000 #same as [HJE18]
iteration_num = 10000

###########################################
#Torch
###########################################

dt = T/(N-1)
X = X_path(N,M,T,xi)
dW = X[0]
X_p = X[1]
E_X = torch.mean(X_p,1,True) #computes the mean of each row
P = torch.zeros((M,N-1)) #initialize prediction matrix
U_hat = torch.zeros((N,M))

###########################################
#Stack of Neural Networks
###########################################
class Net(torch.nn.Module):
        def __init__(self, layer_size):
            super().__init__()
            self.fc1 = torch.nn.Linear(1, layer_size)
            self.fc2 = torch.nn.Linear(layer_size, layer_size)
            self.fc3 = torch.nn.Linear(layer_size, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x

for n in range(N-2,-1,-1): #backwards iteration from N-1 to 0
    x = torch.reshape(X_p[n],(M,1)) #turn column vec into row vec

    
    y = sigma(grad_u(x,dt*n)) #this is cheating. We are not given what grad_u is, but we approximate it by updating the gradients based off the loss function that uses the forward iteration based off th
    
    
    
    #y = g(X_p[N-1],T) #Need to use this for the loss function

    net = Net(layer_size=layer_size)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) #we use Adam

    start_time = time.time()
    for num in range(iteration_num):
        prediction = net(x) # input x and predict based on x
        loss = F.mse_loss(prediction, y) # Not the real loss function from [HJE18]
        optimizer.zero_grad() # clear gradients for next train
        if (num > iteration_num/2):                                 #same learning rate condition as [HJE18] if it_num = 40000 and lr = .01
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate/10)
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients

    P[:,n] = (prediction.detach()).squeeze(1) #note that .detach() removes autograd object from the tensor
    
    print(f"Training for subnetwork {n+1} of {N-1} completed in {round(time.time()-start_time,2)} s. Estimated time remaining: {round((time.time()-start_time)*(n),2)} s.")

###########################################
#Plot for Solution built from from NN Gradient
###########################################

t = np.linspace(0,T,N)

u_0_xi_hat = (compute_u_0_X0(N,M,T,dW, X_p, torch.t(P))).item()

print(f"u(0,xi) is approximately {u_0_xi_hat}.")

U_hat = pde_solution_approx(N,M,T,xi,dW,X_p,torch.t(P),u_0_xi_hat).numpy()

E_U_hat = U_hat.mean(1)

U = pde_solution_true(N,M,T,xi).numpy() #uses true u(0,xi)

E_U = U.mean(1)

#true_u = np.zeros(N)

for i in range(N):
    plt.plot(t, U_hat[:,i])
#    true_u[i] = solution_u(E_X[i],dt*i)

#plt.plot(t,true_u, c = 'g',lw = 3)
plt.plot(t,E_U, c = 'b',lw = 3)
plt.plot(t,E_U_hat, c = 'c',lw = 3)

plt.grid(True)
plt.show()