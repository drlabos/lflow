# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:42:42 2020

@author: sobol
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

"""
for variable names prefer using chars with integer index, e.g. x1, x2, u1
usage of chars or overlapping names (e.g. x and xx) may result in shit,
during replacement of variables with their values
"""


class LabFunc():
    """
    for unary & binary pass one or more LabFunc
    type basic:
    function - callable string
    derivatives=gradient - dictionary with partial derivatives
    """
    def __init__(*fargs, derivatives=None, args=None, theta=1):
        if (len(fargs)==3) and isinstance(fargs[1], str) and isinstance(fargs[2], dict):
            """
            e.g.
            LabFunc('3*x1', {'x1' : 1})
            second argument will be treated as derivatives
            """
            fargs[0].type = 'basic'
            derivatives = derivatives or fargs[2]
            if args:
                fargs[0].args = set(args)
            elif derivatives:
                fargs[0].args = set(derivatives.keys())
            else:
                fargs[0].args = set()
            fargs[0].fun = str(fargs[1])
        elif len(fargs) > 2 and isinstance(fargs[1], LabFunc):
            """
            e.g.
            LabFunc(f1, f2, f3)
            """
            fargs[0].type = 'multi'
            fargs[0].args = fargs[1].args
            for fun in fargs[2:]:
                fargs[0].args = fargs[0].args.union(fun.args)
            fargs[0].fun = list(fargs[1:])
        elif len(fargs) == 2 and isinstance(fargs[1], LabFunc):
            """
            e.g.
            LabFunc(f1), -f1
            """
            fargs[0].type = 'unary'
            fargs[0].args = fargs[1].args
            fargs[0].fun = fargs[1]
        elif isinstance(fargs[1], str) or np.isscalar(fargs[1]):
            """
            LabFunc('3*x1', derivatives={'x1' : 1}), LabFunc(5)
            """
            fargs[0].type = 'basic'
            if args:
                fargs[0].args = set(args)
            elif derivatives:
                fargs[0].args = set(derivatives.keys())
            else:
                fargs[0].args = set()
            fargs[0].fun = str(fargs[1])
        else:
            fargs[0].type = 'basic'
            
        if fargs[0].type == 'basic':
            if '_p_' not in fargs[0].fun:
                for arg in fargs[0].args:
                    fargs[0].fun = fargs[0].fun.replace(arg, '_p_[\'{}\']'.format(arg))
            fargs[0].derivatives = derivatives
            for arg in fargs[0].derivatives:
                if type(fargs[0].derivatives[arg]) != 'str':
                    fargs[0].derivatives[arg] = str(fargs[0].derivatives[arg])
                if '_p_' not in fargs[0].derivatives[arg]:
                    for arg2 in fargs[0].args:
                        fargs[0].derivatives[arg] = fargs[0].derivatives[arg].replace(arg2, '_p_[\'{}\']'.format(arg2))
        
        fargs[0].theta = theta
        fargs[0].name = ''
    
    
    def _process_command(self, cmd, x):
        if isinstance(x, Point):
            for key in x.dict:
                cmd = cmd.replace(key, str(x[key]))
        elif isinstance(x, dict):
            for key in x:
                cmd = cmd.replace(key, str(x[key]))
        elif isinstance(x, pd.Series):
            for key in x.keys():
                cmd = cmd.replace(key, str(x[key]))
        elif hasattr(x, '__iter__') and (len(self.args) == len(x)):
            for i, arg in enumerate(self.args):
                cmd = cmd.replace(arg, str(x[i]))
        elif len(self.args) == 1:
            cmd = cmd.replace(list(self.args)[0], str(x))
        return cmd
     
    def __str__(self):
        if self.type == 'basic':
            return '{}({})'.format(self.name, self.fun)
        else:
            return str(self.fun)
    
    def __repr__(self):
        return str(self)

    
    def _process_arg(self, _p_):
        if isinstance(_p_, Point):
            return _p_, _p_._size
        elif isinstance(_p_, pd.Series):
            return _p_, _p_.shape[0]
        elif isinstance(_p_, dict):
            for k in _p_.keys():
                size = 1 if np.isscalar(_p_[k]) else _p_[k].shape[0]
                break
            return _p_, size
        else:
            """
            vector passed, try to match variables
            may produce unexpected behaviour
            but i leave this case to ease the usage of simpliest functions
            """
            tmp = _p_
            _p_ = {}
            if (len(self.args) == 1) and (np.isscalar(tmp)):
                _p_[list(self.args)[0]] = tmp
            elif hasattr(tmp, '__iter__') and (len(self.args) == len(tmp)):
                for i, arg in enumerate(self.args):
                    _p_[arg] = tmp[i]
            return _p_, 1
        
        
    
    def __call__(self, _p_):
        if self.type == 'basic':
            _p_, size = self._process_arg(_p_)
            res = eval(self.fun)
            if (size>1) and np.isscalar(res):
                return res*np.ones((size, 1))
            else:
                return res
        else:
            raise NotImplementedError
        
    
    def deriv(self, _p_, args=None):
        if self.type == 'basic':
            result = {}
            _p_, size = self._process_arg(_p_)
            keys = args or self.derivatives.keys()
            for arg in keys:
                if arg not in self.derivatives:
                    result[arg] = np.zeros((size, 1))
                else:
                    cmd = self.derivatives[arg]
                    r = eval(cmd)
                    if (size > 1) and np.isscalar(r):
                        result[arg] = r*np.ones((size, 1))
                    else:
                        result[arg] = r
            return Point(result)
        else:
            raise NotImplementedError
    
    
    def __neg__(self):
        return LabNeg(self)
    
    
    def __add__(self, other):
        if isinstance(self, LabSum):
            if isinstance(other, LabSum):
                arg = self.fun + other.fun
                return LabSum(*arg)
            elif isinstance(other, LabFunc):
                arg = self.fun + [other]
                return LabSum(*arg)
            elif np.isscalar(other):
                arg = self.fun + [Constant(other)]
                return LabSum(*arg)
        elif isinstance(other, LabSum):
            arg = [self] + other.fun
            return LabSum(*arg)
        elif np.isscalar(other):
            arg = [self, Constant(other)]
            return LabSum(*arg)
        else:
            arg = [self, other]
            return LabSum(*arg)
        
    def __radd__(self, other):
        return self + other
            
    
    def __sub__(self, other):
        if isinstance(other, LabFunc):
            return LabSubtract(self, other)
        elif np.isscalar(other):
            return LabSubtract(self, Constant(other))

    
    def __rsub__(self, other):
        tmp = -self
        return other + tmp
    
    
    def __mul__(self, other):
        if isinstance(self, LabProd):
            if isinstance(other, LabProd):
                arg = self.fun + other.fun
                return LabProd(*arg)
            elif isinstance(other, LabFunc):
                arg = self.fun + [other]
                return LabProd(*arg)
            elif np.isscalar(other):
                arg = self.fun + [Constant(other)]
                return LabProd(*arg)
        elif isinstance(other, LabProd):
            arg = [self] + other.fun
            return LabProd(*arg)
        elif np.isscalar(other):
            arg = [self, Constant(other)]
            return LabProd(*arg)
        else:
            arg = [self, other]
            return LabProd(*arg)
        
    def __rmul__(self, other):
        return self*other
        
    
    def __truediv__(self, other):
        if isinstance(other, LabFunc):
            return LabDivide(self, other)
        elif np.isscalar(other):
            return LabDivide(self, Constant(other))

        
    def __rtruediv__(self, other):
        if np.isscalar(other):
            return LabDivide(Constant(other), self)
        
    
    def __pow__(self, other):
        if np.isscalar(other):
            return LabPower(self, Constant(other))
        else:
            return LabPower(self, other)
        
        
    def __rpow__(self, other):
        if np.isscalar(other):
            return LabPower(Constant(other), self)
        else:
            raise NotImplementedError
            
        
class Identity(LabFunc):
    """
    identity fuction
    mostly used for variables
    """
    def __init__(self, arg):
        super().__init__(arg, derivatives={arg : 1})
        

class Constant(LabFunc):
    def __init__(self, val):
        super().__init__(str(val), derivatives={}, args=[])
        

class LabNeg(LabFunc):
    def __call__(self, p):
        return -self.fun(p)
    
    def deriv(self, p, args=None):
        return -self.fun.deriv(p, args=args)
    
    def __str__(self):
        return '-{}'.format(self.fun)
    
    def __repr__(self):
        return str(self)

class LabSum(LabFunc):       
    def __call__(self, p):
        p, size = self._process_arg(p)
        if size == 1:
            return np.sum([f(p) for f in self.fun])
        else:
            vals = np.concatenate([f(p).reshape((size, 1)) for f in self.fun], axis=1)
            return np.sum(vals, axis=1, keepdims = True)
    
    def deriv(self, p, args=None):
        res = self.fun[0].deriv(p, args=args)
        for f in self.fun[1:]:
            res += f.deriv(p, args=args)
        return res
    
    def __str__(self):
        return 'sum({})'.format(', '.join([str(f) for f in self.fun]))
    
    def __repr__(self):
        return str(self)
    

class LabSubtract(LabFunc):
    def __call__(self, p):
        return self.fun[0](p) - self.fun[1](p)
    
    def deriv(self, p, args=None):
        return self.fun[0].deriv(p, args=args) - self.fun[1].deriv(p, args=args)
    
    def __str__(self):
        return '{} - {}'.format(self.fun[0], self.fun[1])
    
    def __repr__(self):
        return str(self)
    

class LabProd(LabFunc):
    def __call__(self, p):
        p, size = self._process_arg(p)
        if size == 1:
            return np.prod([f(p) for f in self.fun])
        else:
            vals = np.concatenate([f(p).reshape((size, 1)) for f in self.fun], axis=1)
            return np.prod(vals, axis=1, keepdims = True)
        #return np.prod([f(p) for f in self.fun], axis=0)
    
    def deriv(self, p, args=None):
        p, size = self._process_arg(p)
        keys = args or self.args
        res = Point({arg : 0 for arg in keys})
        big_derivs = []
        for f in self.fun:
            big_derivs.append(f.deriv(p))
        if size==1:
            vals = np.array([f(p) for f in self.fun])
            for arg in keys:
                arg_derivs = []
                for d in big_derivs:
                    if arg in d.dict:
                        arg_derivs.append(d[arg])
                    else:
                        arg_derivs.append(0)
                arg_derivs = np.array(arg_derivs)
                for i in range(len(self.fun)):
                    tmp = vals.copy()
                    tmp[i] = arg_derivs[i]
                    res[arg] = res[arg] + np.prod(tmp)                
        else:
            vals = np.concatenate([f(p).reshape((size, 1)) for f in self.fun], axis=1)
            zero_col = np.zeros((size, 1))
            for arg in keys:
                arg_derivs = []
                for d in big_derivs:
                    if arg in d.dict:
                        arg_derivs.append(d[arg].reshape((size, 1)))
                    else:
                        arg_derivs.append(zero_col)
                arg_derivs = np.concatenate(arg_derivs, axis=1)
                for i in range(len(self.fun)):
                    tmp = vals.copy()
                    tmp[:, i] = arg_derivs[:,i]
                    res[arg] = res[arg] + np.prod(tmp, axis=1, keepdims=True)
        return res
    
    def __str__(self):
        return 'prod({})'.format(', '.join([str(f) for f in self.fun]))
    
    def __repr__(self):
        return str(self)
    

class LabDivide(LabFunc):
    def __call__(self, p):
        return np.divide(self.fun[0](p), self.fun[1](p))
    
    def deriv(self, p, args=None):
        return (self.fun[0].deriv(p, args=args)*self.fun[1](p)  - self.fun[1].deriv(p, args=args)*self.fun[0](p))/(self.fun[1](p)**2)
        if self.fun[1](p) != 0:
            return (self.fun[0].deriv(p, args=args)*self.fun[1](p)  - self.fun[1].deriv(p, args=args)*self.fun[0](p))/(self.fun[1](p)**2)
        else:
            """
            create zero point with same coordinates
            """
            res = self.fun[0].deriv(p, args=args) + self.fun[1].deriv(p, args=args)
            for key in res._keys:
                res[key] = np.nan
            return res
        
    def __str__(self):
        return '{} / {}'.format(self.fun[0], self.fun[1])
    
    def __repr__(self):
        return str(self)
    

class LabPower(LabFunc):
    def __call__(self, p):
        return np.power(self.fun[0](p), self.fun[1](p))
    
    def deriv(self, p, args=None):
        if isinstance(self.fun[0], Constant) and isinstance(self.fun[1], LabFunc):
            """
            c^f2(x) -> f2'(x) * log(c) * c^f2(x)
            """
            return self.fun[1].deriv(p, args=args)*np.log(self.fun[0](p))*self(p)
        elif (not isinstance(self.fun[0], Constant)) and isinstance(self.fun[1], Constant):
            """
            f1(x)^c -> c * f1'(x) * f1(x)^c-1
            """
            return self.fun[0].deriv(p, args=args) * self.fun[1](p) * np.power(self.fun[0](p), self.fun[1](p) - 1)
        else:
            part1 = self.fun[0].deriv(p, args=args) * self.fun[1](p) / self.fun[0](p)
            part2 = self.fun[1].deriv(p, args=args) * np.log(self.fun[0](p))
            return (part1+part2)*self(p)
            
    def __str__(self):
        return 'power({}, {})'.format(self.fun[0], self.fun[1])
    
    def __repr__(self):
        return str(self)
    
    
class LabExp(LabFunc):
    def __call__(self, p):
        return np.exp(self.fun(p))
    
    def deriv(self, p, args=None):
        return self.fun.deriv(p, args=args)*np.exp(self.fun(p))
    
    def __str__(self):
        return 'exp({})'.format(self.fun)
    
    def __repr__(self):
        return str(self)
    

class LabLog(LabFunc):
    def __call__(self, p):
        return np.log(self.fun(p))
    
    def deriv(self, p, args=None):
        return self.fun.deriv(p, args=args)/self.fun(p)
    
    def __str__(self):
        return 'log({})'.format(self.fun)
    
    def __repr__(self):
        return str(self)
    
        
class LabCos(LabFunc):
    def __call__(self, p):
        return np.cos(self.fun(p))
    
    def deriv(self, p, args=None):
        return -self.fun.deriv(p, args=args)*np.sin(self.fun(p))
    
    def __str__(self):
        return 'cos({})'.format(self.fun)
    
    def __repr__(self):
        return str(self)
    
        
class LabSin(LabFunc):
    def __call__(self, p):
        return np.sin(self.fun(p))
    
    def deriv(self, p, args=None):
        return self.fun.deriv(p, args=args)*np.cos(self.fun(p))
    
    def __str__(self):
        return 'sin({})'.format(self.fun)
    
    def __repr__(self):
        return str(self)
        

class LabArctg(LabFunc):
    def __call__(self, p):
        return np.arctan(self.fun(p))
    
    def deriv(self, p, args=None):
        return self.fun.deriv(p, args=args)/(1 + np.power(self.fun(p), 2))
    
    
    def __str__(self):
        return 'arctg({})'.format(self.fun)
    
    def __repr__(self):
        return str(self)
        


class LabSigmoid(LabFunc):
    def __call__(self, p):
        #return np.divide(1, 1+np.exp(-self.theta*self.fun(p)))
        return expit(self.theta*self.fun(p))
    
    def deriv(self, p, args=None):
        return self.theta*self.fun.deriv(p, args=args)*self(p)*(1-self(p))
    
    def __str__(self):
        return 'sigmoid({})'.format(self.fun)
    
    def __repr__(self):
        return str(self)
    
    

class LabIndicator(LabFunc):
    def __call__(self, p):
        return np.heaviside(self.fun(p), 1)
    
    def deriv(self, p, **kwargs):
        return 0
    
    def __str__(self):
        return 'I({} > 0)'.format(self.fun)
    
    def __repr__(self):
        return str(self)
    
        

class LabMax(LabFunc):
    def __call__(self, p):
        if isinstance(self.fun, list):
            p, size = self._process_arg(p)
            if size == 1:
                return max([f(p) for f in self.fun])
            else:
                return np.max(np.concatenate([f(p) for f in self.fun], axis=1), axis=1, keepdims=True)
        else:
            return self.fun(p)
        
    def __str__(self):
        return 'max({})'.format(', '.join([str(f) for f in self.fun]))
    
    def __repr__(self):
        return str(self)
    

class LabMin(LabFunc):
    def __call__(self, p):
        if isinstance(self.fun, list):
            p, size = self._process_arg(p)
            if size == 1:
                return min([f(p) for f in self.fun])
            else:
                return np.min(np.concatenate([f(p) for f in self.fun], axis=1), axis=1, keepdims=True)
        else:
            return self.fun(p)
        
    def __str__(self):
        return 'min({})'.format(', '.join([str(f) for f in self.fun]))
    
    def __repr__(self):
        return str(self)
        

class LabSmoothmax(LabFunc):
    """
    use a little trick here
    I do not calc the derivative of smooth maximum
    I use smooth version of max derivative
    """
    def __call__(self, p):
        p, size = self._process_arg(p)
        if size == 1:
            vals_exp = np.sum([np.exp(self.theta*f(p)) for f in self.fun])
            vals_fexp = np.sum([f(p)*np.exp(self.theta*f(p)) for f in self.fun])
            return vals_fexp/vals_exp
        else:
            vals_exp = np.concatenate([np.exp(self.theta*f(p)) for f in self.fun], axis=1)
            vals_fexp = np.concatenate([f(p)*np.exp(self.theta*f(p)) for f in self.fun], axis=1)
            return np.sum(vals_fexp, axis=1, keepdims=True)/np.sum(vals_exp, axis=1, keepdims=True)
    
    def deriv(self, p, args=None):
        p, size = self._process_arg(p)
        res = {}
        if size==1:
            vals_exp = sum([np.exp(self.theta*f(p)) for f in self.fun])
            vals_fexp = self.fun[0].deriv(p, args=args)*np.exp(self.theta*self.fun[0](p))
            for f in self.fun[1:]:
                vals_fexp = vals_fexp + f.deriv(p, args=args)*np.exp(self.theta*f(p))
            return vals_fexp/vals_exp
        else:
            keys = args or self.args
            vals_exp = np.sum(np.concatenate([np.exp(self.theta*f(p)) for f in self.fun], axis=1), axis=1, keepdims=True)
            for key in keys:
                vals_fexp = np.concatenate([f.deriv(p, args=[key])[key]*np.exp(self.theta*f(p)) for f in self.fun], axis=1)
                res[key] = np.sum(vals_fexp, axis=1, keepdims=True)/vals_exp
            return Point(res)
        
    def __str__(self):
        return 'Smoothmax({})'.format(', '.join([str(f) for f in self.fun]))
    
    def __repr__(self):
        return str(self)
    
        

class LabSmoothmin(LabFunc):
    """
    use a little trick here
    I do not calc the derivative of smooth maximum
    I use smooth version of max derivative
    """
    def __call__(self, p):
        p, size = self._process_arg(p)
        if size == 1:
            vals_exp = np.sum([np.exp(-self.theta*f(p)) for f in self.fun])
            vals_fexp = np.sum([f(p)*np.exp(-self.theta*f(p)) for f in self.fun])
            return vals_fexp/vals_exp
        else:
            vals_exp = np.concatenate([np.exp(-self.theta*f(p)) for f in self.fun], axis=1)
            vals_fexp = np.concatenate([f(p)*np.exp(-self.theta*f(p)) for f in self.fun], axis=1)
            return np.sum(vals_fexp, axis=1, keepdims=True)/np.sum(vals_exp, axis=1, keepdims=True)
    
    def deriv(self, p, args=None):
        p, size = self._process_arg(p)
        res = {}
        if size==1:
            vals_exp = sum([np.exp(-self.theta*f(p)) for f in self.fun])
            vals_fexp = self.fun[0].deriv(p, args=args)*np.exp(-self.theta*self.fun[0](p))
            for f in self.fun[1:]:
                vals_fexp = vals_fexp + f.deriv(p, args=args)*np.exp(-self.theta*f(p))
            return vals_fexp/vals_exp
        else:
            keys = args or self.args
            vals_exp = np.sum(np.concatenate([np.exp(-self.theta*f(p)) for f in self.fun], axis=1), axis=1, keepdims=True)
            for key in keys:
                vals_fexp = np.concatenate([f.deriv(p, args=[key])[key]*np.exp(-self.theta*f(p)) for f in self.fun], axis=1)
                res[key] = np.sum(vals_fexp, axis=1, keepdims=True)/vals_exp
            return Point(res)
        
    def __str__(self):
        return 'Smoothmin({})'.format(', '.join([str(f) for f in self.fun]))
    
    def __repr__(self):
        return str(self)
    



if __name__ == '__main__':
    from labos_point import Point
    p1 = Point({'x1' : 1, 'x2' : 1})
    p2 = Point({'x1' : 2, 'x2' : 3})
    f1 = LabFunc('3*x1+2', derivatives={'x1' : '3'})
    f2 = LabFunc('np.sin(x1)', derivatives={'x1' : 'np.cos(x1)'})
    f3 = LabFunc('x1**2 - 2*x1*x2', derivatives={'x1': '2*x1 - 2*x2', 'x2' : '-2*x2'})
    t2 = LabExp(f1)
    t3 = LabLog(t2)
    t4 = LabCos(f2+f3)
    t5 = LabSin(f1+f2)
    t6 = LabSigmoid(f1+f2-f3)
    smax = LabSmoothmax(f1, f2, theta=10)
    smin = LabSmoothmin(f1, f2, theta=10)
    xs = np.arange(-3, 3, 0.01)
    ###