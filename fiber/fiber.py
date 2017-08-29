'''
Small class to handle a single fiber. The two main references are
1.
2.

We use a linearized implicit penalty method to enforce inextensibility

X_s * X_st = (\tau / dt) * (1 - X^n * X^{n+1})

We use a local mobility derived from slender body theory, see Ref 1 or 2.

M = c_0 * (1 + X_s * X_s) + c_1 * (1 - X_s * X_2)

with
c_0 = -log(e * epsilon**2) / (8 * pi * viscosity)
c_1 = 2 / (8 * pi * viscosity)
'''
# Standard imports
import numpy as np
import scipy.linalg
from functools import partial
import sys
sys.path.append('../')

# Local imports
from utils import cheb


class fiber(object):
  '''
  Small class to handle a single fiber.
  '''
  def __init__(self, 
               num_points = 32, 
               length = 1.0, 
               epsilon = 1e-03, 
               E = 1.0, 
               viscosity = 1.0, 
               dt = 1e-03, 
               BC_start_0 = 'force',
               BC_start_1 = 'torque',
               BC_end_0 = 'force',
               BC_end_1 = 'torque',
               BC_start_vec_0 = np.zeros(3),
               BC_start_vec_1 = np.zeros(3),
               BC_end_vec_0 = np.zeros(3),
               BC_end_vec_1 = np.zeros(3)):
    # Store some parameters
    self.num_points = num_points
    self.length = length
    self.epsilon = epsilon
    self.E = E
    self.viscosity = viscosity
    self.dt = dt
    self.c_0 = -np.log(np.e * epsilon**2) / (8.0 * np.pi * viscosity)
    self.c_1 = 2.0 / (8.0 * np.pi * viscosity) 
    self.beta = -np.log(np.e * epsilon**2) * np.maximum(E, 10.0) / (length**4 * dt)  
    self.BC_start_0 = BC_start_0
    self.BC_start_1 = BC_start_1
    self.BC_end_0 = BC_end_0
    self.BC_end_1 = BC_end_1
    self.BC_start_vec_0 = BC_start_vec_0
    self.BC_start_vec_1 = BC_start_vec_1
    self.BC_end_vec_0 = BC_end_vec_0
    self.BC_end_vec_1 = BC_end_vec_1

    print 'beta = ', self.beta

    # Get Chebyshev differential matrix
    D_1, s = cheb.cheb(num_points - 1)

    # Flip material coordinate to go from -1 to 1
    self.s = np.flipud(s)
    self.D_1_0 = np.flipud(np.flipud(D_1.T).T)

    # Get high order derivatives, enforce D * vector_ones = 0
    self.D_2_0 = np.dot(self.D_1_0, self.D_1_0)
    self.D_2_0 -= np.diag(np.sum(self.D_2_0.T, axis=0))
    self.D_3_0 = np.dot(self.D_2_0, self.D_1_0)
    self.D_3_0 -= np.diag(np.sum(self.D_3_0.T, axis=0))
    self.D_4_0 = np.dot(self.D_2_0, self.D_2_0)
    self.D_4_0 -= np.diag(np.sum(self.D_4_0.T, axis=0))

    # Differential matrices for any length
    self.D_1 = self.D_1_0 * (2.0 / self.length)
    self.D_2 = self.D_2_0 * (2.0 / self.length)**2
    self.D_3 = self.D_3_0 * (2.0 / self.length)**3
    self.D_4 = self.D_4_0 * (2.0 / self.length)**4
    
    # Create fiber configuration, straight fiber along x
    self.x = np.zeros((num_points, 3))
    self.x[:,0] = -s / 2.0
    return

  def scale_D():
    '''
    Scale differential matrices with fiber length. Note that Chebyshev polynomial
    assumes a domain of length 2 ([-1, 1]).
    '''
    D_1 = D_1_0 * (2.0 / self.length)
    D_2 = D_2_0 * (2.0 / self.length)**2
    D_3 = D_3_0 * (2.0 / self.length)**3
    D_4 = D_4_0 * (2.0 / self.length)**4
    
  def set_conf(self, x):
    '''
    Set fiber configuration. It does not check if inextensibility holds.
    '''
    self.x = x
    return


  def set_BC(self,
             BC_start_0 = 'force',
             BC_start_1 = 'torque',
             BC_end_0 = 'force',
             BC_end_1 = 'torque',
             BC_start_vec_0 = np.zeros(3),
             BC_start_vec_1 = np.zeros(3),
             BC_end_vec_0 = np.zeros(3),
             BC_end_vec_1 = np.zeros(3)):    
    '''
    Set Boundary Conditions options. For each end of the fiber
    (labeled start and end) we need to provide two boundary conditions,
    one for the translational degrees of freedom and other for the orientations.
    
    options for translation:
    force = it applies a force to the end of the fiber.
    position = it prescribes the position of the fiber's end.
    
    options for rotation:
    torque = it applies a torque to the end of the fiber.
    angle = it enforces an orientation of the fiber's end.

    '''
    self.BC_start_0 = BC_start_0
    self.BC_start_1 = BC_start_1
    self.BC_end_0 = BC_end_0
    self.BC_end_1 = BC_end_1
    self.BC_start_vec_0 = BC_start_vec_0
    self.BC_start_vec_1 = BC_start_vec_1
    self.BC_end_vec_0 = BC_end_vec_0
    self.BC_end_vec_1 = BC_end_vec_1
    return
    

  def form_linear_operator(self):
    '''
    Returns the linear operator A that define the linear system
    
    A * (X^{n+1}, T^{n+1}) = RHS    
    '''
    # Compute material derivatives at time 
    xs = np.dot(self.D_1, self.x) 
    xss = np.dot(self.D_2, self.x) 
    xsss = np.dot(self.D_3, self.x) 
    xssss = np.dot(self.D_4, self.x) 
    
    # Allocate memory for matrix
    A = np.zeros((4 * self.num_points, 4 * self.num_points))

    I = np.eye(self.num_points)

    # Build submatrices to couple coordinates to coordinates
    A_XX = (I / self.dt) \
           + self.E * self.c_0 * (np.dot((I + np.diag(xs[:,0]**2)), self.D_4)) \
           + self.E * self.c_1 * (np.dot((I - np.diag(xs[:,0]**2)), self.D_4))
    A_XY = self.E * self.c_0 * np.dot(np.diag(xs[:,0]*xs[:,1]), self.D_4) - self.E * self.c_1 * np.dot(np.diag(xs[:,0]*xs[:,1]), self.D_4) 
    A_XZ = self.E * self.c_0 * np.dot(np.diag(xs[:,0]*xs[:,2]), self.D_4) - self.E * self.c_1 * np.dot(np.diag(xs[:,0]*xs[:,2]), self.D_4) 
    A_YY = (I / self.dt) \
           + self.E * self.c_0 * np.dot((I + np.diag(xs[:,1]**2)), self.D_4) \
           + self.E * self.c_1 * np.dot((I - np.diag(xs[:,1]**2)), self.D_4) 
    A_YZ = self.E * self.c_0 * np.dot(np.diag(xs[:,1]*xs[:,2]), self.D_4) - self.E * self.c_1 * np.dot(np.diag(xs[:,1]*xs[:,2]), self.D_4)
    A_ZZ = (I / self.dt) \
           + self.E * self.c_0 * np.dot((I + np.diag(xs[:,2]**2)), self.D_4) \
           + self.E * self.c_1 * np.dot((I - np.diag(xs[:,2]**2)), self.D_4)

    # Build submatrices to couple tension to coordinates
    A_XT = -self.c_0 * (2.0 * np.dot(np.diag(xs[:,0]), self.D_1) + np.diag(xss[:,0])) - self.c_1 * np.diag(xss[:,0])
    A_YT = -self.c_0 * (2.0 * np.dot(np.diag(xs[:,1]), self.D_1) + np.diag(xss[:,1])) - self.c_1 * np.diag(xss[:,1])
    A_ZT = -self.c_0 * (2.0 * np.dot(np.diag(xs[:,2]), self.D_1) + np.diag(xss[:,2])) - self.c_1 * np.diag(xss[:,2])

    # Build submatrices coordinates to tension
    c = np.log(np.e * self.epsilon**2);
    A_TX = self.E * 6.0 * c * np.dot(np.diag(xsss[:,0]), self.D_3) - \
           self.E * (2.0 - 7.0 * c) * np.dot(np.diag(xss[:,0]), self.D_4) - \
           self.beta * np.dot(np.diag(xs[:,0]), self.D_1)
    A_TY = self.E * 6.0 * c * np.dot(np.diag(xsss[:,1]), self.D_3) - \
           self.E * (2.0 - 7.0 * c) * np.dot(np.diag(xss[:,1]), self.D_4) - \
           self.beta * np.dot(np.diag(xs[:,1]), self.D_1)
    A_TZ = self.E * 6.0 * c * np.dot(np.diag(xsss[:,2]), self.D_3) - \
           self.E * (2.0 - 7.0 * c) * np.dot(np.diag(xss[:,2]), self.D_4) - \
           self.beta * np.dot(np.diag(xs[:,2]), self.D_1)
    
    # Build submatrices tension to tension
    A_TT = 2.0 * c * self.D_2 + (2.0 - c) * np.diag(xss[:,0]**2  + xss[:,1]**2 + xss[:,2]**2)

    # Collect all block matrices
    A = np.vstack((np.hstack((A_XX, A_XY, A_XZ, A_XT)),
                   np.hstack((A_XY, A_YY, A_YZ, A_YT)),
                   np.hstack((A_XZ, A_YZ, A_ZZ, A_ZT)),
                   np.hstack((A_TX, A_TY, A_TZ, A_TT))))    
    return A


  def compute_RHS(self, force_external = None, flow = None):
    '''
    Compute the Right Hand Side for the linear system
    A * (X^{n+1}, T^{n+1}) = RHS 

    with
    RHS = (X^n / dt + flow + Mobility * force_external, ...)
    
    Note that the internal force contributions (flexibility and in extensibility) 
    force_internal includes flexibility and inextensibility contributions have been
    included in the linear operator A.
    '''
    # Compute material derivatives at time n
    xs = np.dot(self.D_1, self.x) 

    I = np.eye(self.num_points)

    # Build RHS
    RHS = np.zeros(4 * self.num_points)
    RHS[0:self.num_points]                   = self.x[:,0] / self.dt 
    RHS[self.num_points:2*self.num_points]   = self.x[:,1] / self.dt 
    RHS[2*self.num_points:3*self.num_points] = self.x[:,2] / self.dt 
    RHS[3*self.num_points:]                  = -self.beta 

    # Add background flow contribution
    if flow is not None:
      RHS[0:self.num_points]                   += flow[:,0]
      RHS[self.num_points:2*self.num_points]   += flow[:,1]
      RHS[2*self.num_points:3*self.num_points] += flow[:,2]
      RHS[3*self.num_points:] -= (8.0 * np.pi * self.viscosity) * (xs[:,0] * np.dot(self.D_1, flow[:,0]) +
                                                                   xs[:,1] * np.dot(self.D_1, flow[:,1]) +
                                                                   xs[:,2] * np.dot(self.D_1, flow[:,2]))

    # Add external force contribution 
    if force_external is not None: 
      xss = np.dot(self.D_2, self.x) 
      fs = np.dot(self.D_1, force_external)
      RHS[0:self.num_points] += self.c_0 * np.dot((I + np.diag(xs[:,0]**2)),         force_external[:,0]) + \
                                self.c_0 * np.dot((                          np.diag(xs[:,0] * xs[:,1])),  force_external[:,1]) + \
                                self.c_0 * np.dot((                          np.diag(xs[:,0] * xs[:,2])),  force_external[:,2]) + \
                                self.c_1 * np.dot((I - np.diag(xs[:,0]**2)),         force_external[:,0]) + \
                                self.c_1 * np.dot((                        - np.diag(xs[:,0] * xs[:,1])),  force_external[:,1]) + \
                                self.c_1 * np.dot((                        - np.diag(xs[:,0] * xs[:,2])),  force_external[:,2]) 
      RHS[self.num_points:2*self.num_points] += self.c_0 * np.dot((                          np.diag(xs[:,1] * xs[:,0])), force_external[:,0]) + \
                                                self.c_0 * np.dot((I + np.diag(xs[:,1]**2)),        force_external[:,1]) + \
                                                self.c_0 * np.dot((                          np.diag(xs[:,1] * xs[:,2])), force_external[:,2]) + \
                                                self.c_1 * np.dot((                        - np.diag(xs[:,1] * xs[:,0])), force_external[:,0]) + \
                                                self.c_1 * np.dot((I - np.diag(xs[:,1]**2)),        force_external[:,1]) + \
                                                self.c_1 * np.dot((                        - np.diag(xs[:,1] * xs[:,2])), force_external[:,2])     
      RHS[2*self.num_points:3*self.num_points] += self.c_0 * np.dot((                          np.diag(xs[:,2] * xs[:,0])),  force_external[:,0]) + \
                                                  self.c_0 * np.dot((                          np.diag(xs[:,2] * xs[:,1])),  force_external[:,1]) + \
                                                  self.c_0 * np.dot((I + np.diag(xs[:,2]**2)),         force_external[:,2]) + \
                                                  self.c_1 * np.dot((                        - np.diag(xs[:,2] * xs[:,0])),  force_external[:,0]) + \
                                                  self.c_1 * np.dot((                        - np.diag(xs[:,2] * xs[:,1])),  force_external[:,1]) + \
                                                  self.c_1 * np.dot((I - np.diag(xs[:,2]**2)),         force_external[:,2]) 
      RHS[3*self.num_points:] += self.c_0 * (2.0 * (xs[:,0] * fs[:,0] + xs[:,1] * fs[:,1] + xs[:,2] * fs[:,2]) + 
                                             (xss[:,0] * force_external[:,0] + xss[:,1] * force_external[:,1] + xss[:,2] * force_external[:,2])) - \
        self.c_1 * (xss[:,0] * force_external[:,0] + xss[:,1] * force_external[:,1] + xss[:,2] * force_external[:,2])  
    return RHS


  def apply_BC(self, A, RHS):
    '''
    Modify linear operator A and RHS to obey boundary conditions.
    '''
    # Compute material derivatives at time n
    xs = np.dot(self.D_1, self.x) 
    xss = np.dot(self.D_2, self.x) 

    I = np.eye(self.num_points)
    num = self.num_points

    # Apply BC at one end
    if self.BC_start_0 == 'force':
      # Apply force
      A[0,:]               = 0
      A[0,0:num]           = -self.D_3[0,:]
      A[0,3*num:]          = xs[0,0] * I[0,:]
      A[num,:]             = 0
      A[num,num:2*num]     = -self.D_3[0,:]
      A[num,3*num:]        = xs[0,1] * I[0,:]
      A[2*num,:]           = 0
      A[2*num,2*num:3*num] = -self.D_3[0,:]
      A[2*num,3*num:]      = xs[0,2] * I[0,:]
      A[3*num,:]           = 0
      A[3*num,0:num]       = self.D_2[0,:] * xss[0,0]
      A[3*num,num:2*num]   = self.D_2[0,:] * xss[0,1]
      A[3*num,2*num:3*num] = self.D_2[0,:] * xss[0,2]
      A[3*num,3*num:]      = I[0,:]

      RHS[0]     = self.BC_start_vec_0[0]
      RHS[num]   = self.BC_start_vec_0[1]
      RHS[2*num] = self.BC_start_vec_0[2]
      RHS[3*num] = np.dot(self.BC_start_vec_0, xs[0,:]) 
    else:
      # Enforce position
      A[0,:] = 0 
      A[0,0:num]           = I[0,:] / self.dt
      A[num,:] = 0
      A[num,num:2*num]     = I[0,:] / self.dt
      A[2*num,:] = 0
      A[2*num,2*num:3*num] = I[0,:] / self.dt
      A[3*num,:] = 0
      A[3*num,0:num]       = self.E * 3 * xss[0,0] * self.D_3[0,:]
      A[3*num,num:2*num]   = self.E * 3 * xss[0,1] * self.D_3[0,:]
      A[3*num,2*num:3*num] = self.E * 3 * xss[0,2] * self.D_3[0,:]
      A[3*num,3*num:]      = self.D_1[0,:]
      
      RHS[0]     = self.BC_start_vec_0[0] / self.dt
      RHS[num]   = self.BC_start_vec_0[1] / self.dt
      RHS[2*num] = self.BC_start_vec_0[2] / self.dt
      RHS[3*num] = 0
    if self.BC_start_1 == 'torque':
      # Apply torque
      offset = 1
      A[offset,:]                 = 0
      A[offset,0:num]             = self.D_2[offset-1,:]
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = self.D_2[offset-1,:]
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = self.D_2[offset-1,:]

      RHS[offset]       = self.BC_start_vec_1[0]
      RHS[offset+num]   = self.BC_start_vec_1[1]
      RHS[offset+2*num] = self.BC_start_vec_1[2] 
    else:
      # Enforce orientation
      offset = 1
      A[offset,:] = 0
      A[offset,0:num] = self.D_1[offset-1,:]
      A[offset+num,:] = 0
      A[offset+num,num:2*num] = self.D_1[offset-1,:]
      A[offset+2*num,:] = 0
      A[offset+2*num,2*num:3*num] = self.D_1[offset-1,:]  
      
      RHS[offset] = self.BC_start_vec_1[0]
      RHS[offset+num] = self.BC_start_vec_1[1]
      RHS[offset+2*num] = self.BC_start_vec_1[2]

    # Apply BC at the other end
    if self.BC_end_0 == 'force':
      # Apply force
      offset = num - 1
      A[offset,:]                 = 0
      A[offset,0:num]             = self.D_3[offset,:]
      A[offset,3*num:]            = -xs[offset,0] * I[offset,:]
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = self.D_3[offset,:]
      A[offset+num,3*num:]        = -xs[offset,1] * I[offset,:]
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = self.D_3[offset,:]
      A[offset+2*num,3*num:]      = -xs[offset,2] * I[offset,:]
      A[offset+3*num,:]           = 0
      A[offset+3*num,0:num]       = -self.D_2[offset,:] * xss[offset,0]
      A[offset+3*num,num:2*num]   = -self.D_2[offset,:] * xss[offset,1]
      A[offset+3*num,2*num:3*num] = -self.D_2[offset,:] * xss[offset,2]
      A[offset+3*num,3*num:]      = -I[offset,:]

      RHS[offset]       = self.BC_end_vec_0[0]
      RHS[offset+num]   = self.BC_end_vec_0[1]
      RHS[offset+2*num] = self.BC_end_vec_0[2]
      RHS[offset+3*num] = np.dot(self.BC_end_vec_0, xs[offset,:]) 
    else:
      # Enforce position
      offset = num - 1 
      A[offset,:]                 = 0
      A[offset,0:num]             = I[offset,:] / self.dt
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = I[offset,:] / self.dt
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = I[offset,:] / self.dt
      A[offset+3*num,:]           = 0
      A[offset+3*num,0:num]       = self.E * 3 * xss[offset,0] * self.D_3[offset,:]
      A[offset+3*num,num:2*num]   = self.E * 3 * xss[offset,1] * self.D_3[offset,:]
      A[offset+3*num,2*num:3*num] = self.E * 3 * xss[offset,2] * self.D_3[offset,:]
      A[offset+3*num,3*num:]      = self.D_1[offset,:]
      
      RHS[offset]       = self.BC_end_vec_0[0] / self.dt
      RHS[offset+num]   = self.BC_end_vec_0[1] / self.dt
      RHS[offset+2*num] = self.BC_end_vec_0[2] / self.dt
      RHS[offset+3*num] = 0
    if self.BC_end_1 == 'torque':
      # Apply torque
      offset = num - 2
      A[offset,:]                 = 0
      A[offset,0:num]             = self.D_2[offset+1,:]
      A[offset+num,:]             = 0
      A[offset+num,num:2*num]     = self.D_2[offset+1,:]
      A[offset+2*num,:]           = 0
      A[offset+2*num,2*num:3*num] = self.D_2[offset+1,:]

      RHS[offset]       = self.BC_end_vec_1[0] 
      RHS[offset+num]   = self.BC_end_vec_1[1]
      RHS[offset+2*num] = self.BC_end_vec_1[2]
    else:
      # Enforce orientation
      offset = num - 2
      A[offset,:] = 0
      A[offset,0:num] = self.D_1[offset+1,:]
      A[offset+num,:] = 0
      A[offset+num,num:2*num] = self.D_1[offset+1,:]
      A[offset+2*num,:] = 0
      A[offset+2*num,2*num:3*num] = self.D_1[offset+1,:]  
      
      RHS[offset] = self.BC_end_vec_1[0]
      RHS[offset+num] = self.BC_end_vec_1[1]
      RHS[offset+2*num] = self.BC_end_vec_1[2]
    return A, RHS

  

