# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:43:04 2021

@author: sobol
делаем обертки для функции вероятности и функции квантили
с соответствующими производными
"""

from labos_flow_v2 import np, plt, Identity, LabSigmoid, Point, LabFunc, LabSin
from stochastic_model import IndependentGenerator, StochGenerator
from scipy import stats

class ProbabilityFunction(LabFunc):
    def __init__(self, phux, generator, phi_name = 'phi', theta=10, sample_size=5e3):
        if not isinstance(phux, LabFunc):
            raise ValueError('first argument must be of type LabFunc')
        if not isinstance(generator, StochGenerator):
            raise ValueError('second argument must be of type StochGenerator')
            
        self.phux = phux
        self.generator = generator
        self.phi = Identity(phi_name)
        self.phi_name = phi_name
        self.theta = theta
        self.x_args = list(generator.args)
        self.u_args = list(filter(lambda u: u not in generator.args, phux.args))
        _, self._sample = generator.rvs(int(sample_size))
        self.fun = LabSigmoid(self.phi - self.phux, theta=self.theta)
        
        
    def __call__(self, point, phi):
        p = point.expand({self.phi_name : phi})
        return self.generator.papa_carlo(self.fun, p, self._sample)
    
    
    def deriv(self, point, phi, args=None, phi_derivative=False):
        args = args or self.u_args
        if phi_derivative:
            args = args + [self.phi_name]
        p = point.expand({self.phi_name : phi})
        val, grad = self.generator.papa_carlo(self.fun, p, self._sample, derivs=args)
        return grad, val


class QuantileFunction(LabFunc):
    def __init__(*args, **kwargs):
        """
        two choices: pass ProbabilityFunction directly, to create linked object
        or pass phux and generator as arguments, theta and sample_size as keywords
        additional keyword arguments
        alpha - required probability
        phi_min - used as lower estimate of credible phi values while estimating quantile
        phi_max - used as upper estimate of credible phi values while estimating quantile
        phi_tol - used to identify required accuracy
        """
        if isinstance(args[1], ProbabilityFunction):
            args[0].pphi = args[1]
        else:
            args[0].pphi = ProbabilityFunction(args[1], args[2], **kwargs)
        args[0].x_args = args[0].pphi.x_args
        args[0].u_args = args[0].pphi.u_args
        if 'alpha' in kwargs:
            args[0].alpha = kwargs['alpha']
        else:
            print('Required probability is not set. Default value if 0.9')
            args[0].alpha = 0.9
        """
        args[0].phi_tol = kwargs.get('phi_tol') or 1e-3
        args[0].phi_min = kwargs.get('phi_min')
        args[0].phi_max = kwargs.get('phi_max')
        """
        
        
    def __call__(self, point):
        """
        solving equation self.pphi = self.alpha
        numpy uses linear interpolation by default
        """
        _, vals = self.pphi.generator.papa_carlo(self.pphi.phux, point, self.pphi._sample, return_vals=True)
        return np.quantile(vals, self.alpha)
    
    
    def deriv(self, point, args=None):
        args = args or self.u_args
        q = self(point)
        grad, _ = self.pphi.deriv(point, q, args=args, phi_derivative=True)
        for arg in args:
            grad[arg] = grad[arg]/grad[self.pphi.phi_name]
        return grad
    
    
                
            
        


if __name__ == '__main__':
    u1 = Identity('u1')
    u2 = Identity('u2')
    X1 = Identity('X1')
    X2 = Identity('X2')
    phux = u1*X1 + LabSin(u2)*X2
    generator = IndependentGenerator(['X1', 'X2'], [stats.uniform(), stats.uniform(loc=2, scale=3)])
    
    pphi = ProbabilityFunction(phux, generator)
    pnt = Point({'u1' : 2, 'u2' : 1})
    p = pphi(pnt, 5)
    grad1, _ = pphi.deriv(pnt, 2)
    grad2, _ = pphi.deriv(pnt, 2, phi_derivative=True)
    
    quantile = QuantileFunction(pphi, alpha=p)
    q = quantile(pnt)
    q_grad = quantile.deriv(pnt, ['u1', 'u2'])
