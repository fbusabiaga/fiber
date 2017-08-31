import numpy as np


def cheb(N):
  '''
  Compute the differentiation matrix D and the Chebyshev extrema points x
  for a polynomial of degree N.

  Inputs:
  N = degree of Chebyshev polynomial
  Outputs:
  D = (N+1) x (N+1) differentiation matrix.
  x = (N+1) extrema points

  Translated cheb.m to python.
  '''
  if N == 0:
    D = 0
    x = 1
    return (D, x)   
  else:
    s = np.linspace(0, N, N+1)
    x = np.reshape(np.cos(np.pi * s / N), (N+1,1))
    c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**s).reshape(N+1,1)
    X = np.tile(x,(1,N+1))
    dX = X - X.T
    D = np.dot(c, 1./c.T) / (dX + np.eye(N+1))
    D -= np.diag(np.sum(D.T, axis=0))    
    return D, x.reshape(N+1)



def clencurt(N):
  '''
  Return weights w for Clenshaw-Curtis quadrature.
  
  Translated from matlab clencurt.m.
  '''
  theta = (np.pi / N) * np.reshape(np.arange(N+1), (N+1,1)) 
  x = np.cos(theta)
  w = np.zeros(N+1)
  ii = np.arange(1, N)
  v = np.ones((N-1,1))

  if (N % 2) == 0:
    w[0] = 1.0 / (N**2 - 1.0) 
    w[N] = w[0]
    for k in range(1, N/2):
      v = v - 2.0 * np.cos(2.0 * k * theta[ii]) / (4.0 * k**2 - 1)
    v = v - np.cos(N * theta[ii]) / (N**2 - 1)
  else:
    w[0] = 1.0 / N**2
    w[N] = w[0]
    for k in range(1, (N-1)/2 + 1):
      v = v - 2.0 * np.cos(2.0 * k * theta[ii]) / (4.0 * k**2 - 1)     
  w[ii] = 2.0 * np.reshape(v, v.size) / N
  return w 



def cheb_extrema_points(N):
  '''
  Compute the Chebyshev extrema points x for a polynomial of degree N.

  Inputs:
  N = degree of Chebyshev polynomial
  Outputs:
  x = (N+1) extrema points
  '''
  if N == 0:
    D = 0
    x = 1
    return x
  else:
    s = np.linspace(0, N, N+1)
    x = np.cos(np.pi * s / N)
    return x



def cheb_calc_coef(x):
  '''
  Compute the Chebyshev coefficients with the values of a function
  evaluated at the Chebyshev extrema points.
  '''
  cheb_coef = np.zeros(x.size)
  c = np.ones(x.size)
  c[0] = 2.0
  c[-1] = 2.0
  for j in range(cheb_coef.size):
    sum = 0.0
    for k in range(cheb_coef.size):
      sum += x[k] * np.cos(np.pi * j * k / (cheb_coef.size - 1)) / c[k]
    cheb_coef[j] = (2.0 / (cheb_coef.size - 1)) * sum / c[j]
  return cheb_coef
  


def cheb_eval(alpha, cheb_coef):
  '''
  Evaluate Chebyshev polynomial at the points x with a naive method.

  TODO: implement Clenshaw's recurrence formula.
  '''
  i = np.arange(alpha.size)
  # print i.shape, alpha.shape, cheb_coef.shape
  x = np.zeros(alpha.size)
  for k in range(cheb_coef.size):
    x += cheb_coef[k] * np.cos(k * np.arccos(alpha))
  return x


