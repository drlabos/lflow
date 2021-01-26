# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:16:39 2020

@author: sobol
"""

from labos_point import Point
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KarmaStrike(Exception):
    pass


class StochGenerator():
    """
    Used for sample generation from different distributions
    3 cases: 
        independent distributions
        multivariate gaussian distribution
        copula
    
    args - sequence of variable names
    """
    def __init__(self, args, **kwargs):
        self.args = [args] if type(args)==str else args

    
    def rvs(self, size):
        """
        sample generation
        """
        pass
    
    
    def _sample_to_point(self, sample):
        df = pd.DataFrame(columns=self.args, data=sample)
        points = [Point(dct) for dct in df.to_dict(orient='records')]
        dicts = df.to_dict(orient='list')
        for key in dicts:
            dicts[key] = np.array(dicts[key]).reshape(df.shape[0], 1)
        return points, dicts
    
    
    def visualize(self, points, args=None, colors=None):
        if args is None:
            args = self.args
            
        if len(args) == 1:
            xs = [point[args[0]] for point in points]
            plt.hist(xs, bins=20)
        else:
            xs = [point[args[0]] for point in points]
            ys = [point[args[1]] for point in points]
            if colors is None:
                plt.scatter(xs, ys)
            else:
                plt.scatter(xs, ys, c=colors)
            
            
    def papa_carlo(self, func, point, sample, derivs=None, return_vals = False):
        """
        calculate mean value of function and its derivative in a given point
        """
        res = 0
        vals = []
        gradient = {}
        if derivs is not None:
            for deriv in derivs:
                if deriv in point.dict:
                    gradient[deriv] = 0
        if isinstance(sample, list):
            """
            list of Points
            """
            if derivs is None:
                for samp in sample:
                    val = func(point.expand(samp))
                    res += val
                    vals.append(val)
                res = res/len(sample)
                return res, vals if return_vals else res
            else:
                for samp in sample:
                    p = point.expand(samp)
                    val = func(p)
                    res += val
                    vals.append(val)
                    grad = func.deriv(p, args=derivs)
                    for deriv in gradient:
                        gradient[deriv] += grad[deriv]
                res = res/len(sample)
                for deriv in gradient:
                    gradient[deriv] = gradient[deriv]/len(sample)
                if return_vals:
                    return res, Point(gradient), vals
                else:
                    return res, Point(gradient)
        elif isinstance(sample, dict):
            """
            it's a crutch!
            but i just belive, that dict contains np.arrays of same shape
            """
            size = sample[list(sample.keys())[0]].shape[0]
            for key in point.dict:
                sample[key] = point[key]*np.ones(shape=(size, 1))
            vals = func(sample)
            res = np.sum(vals)/size
            if derivs is None:
                if return_vals:
                    return res, vals
                else:
                    return res
            else:
                grad = func.deriv(sample, args=derivs)
                for deriv in gradient:
                    gradient[deriv] = np.sum(grad[deriv])/size
                if return_vals:
                    return res, Point(gradient), vals
                else:
                    return res, Point(gradient)
        

class DiscreteVectorGenerator(StochGenerator):
    def __init__(self, args, vectors, probs, **kwargs):
        """
        args - variable names
        vectors - 2d array, each row represent vector value
        probs - probabilities
        """
        super().__init__(args, **kwargs)     
        self.vectors = vectors
        self.probs = probs
        
    
    def rvs(self, size):
        choices = np.random.choice(self.vectors.shape[0], size=size, p=self.probs)
        return self._sample_to_point(self.vectors[choices])
    
    def papa_carlo(self, func, point, sample, derivs=[], return_vals = False):
        """
        sample is redundant, left for compatibility
        calc expectectation directly
        """
        if len(sample) == 0:
            res = 0
            gradient = {}
            for deriv in derivs:
                if deriv in point.dict:
                    gradient[deriv] = 0
            for vi, vector in enumerate(self.vectors):
                p = point.expand({self.args[i] : vector[i] for i in range(len(self.args))})
                res += func(p) * self.probs[vi]
                grad = func.deriv(p) * self.probs[vi]
                for deriv in gradient:
                    gradient[deriv] += grad[deriv]
            if len(derivs) == 0:
                return res
            else:
                return res, Point(gradient)
        else:
            return super().papa_carlo(func, point, sample, derivs=derivs, return_vals=return_vals)
            
            
    
class IndependentGenerator(StochGenerator):
    def __init__(self, args, models, **kwargs):
        """
        models - sequence of scipy.stats._distn_infrastructure.rv_frozen
        like stats.unif(loc=1, scale=2) etc
        """
        super().__init__(args, **kwargs)
        self.models = models if hasattr(models, '__len__') else [models]
        if len(self.args) != len(self.models):
            print('Number of variables must be equal to the number of models')
            
            
    def rvs(self, size):
        res = np.zeros(shape=(size, len(self.args)))
        for i, model in enumerate(self.models):
            res[:,i] = model.rvs(size=size)
        return self._sample_to_point(res)
    
    
class MultivariateGaussGenerator(StochGenerator):
    def __init__(self, args, **kwargs):
        """
        models - sequence of scipy.stats._distn_infrastructure.rv_frozen
        like stats.unif(loc=1, scale=2) etc
        """
        super().__init__(args, **kwargs)
        mean = kwargs.get('mean') or np.zeros(shape=(len(self.args)))
        cov = kwargs.get('cov') or np.eye(len(self.args))
        self.model = stats.multivariate_normal(mean=mean, cov=cov)
            
            
    def rvs(self, size):
        return self._sample_to_point(self.model.rvs(size=size))
    
    
if __name__ == '__main__':
    mvn1 = MultivariateGaussGenerator(['x1', 'x2', 'x3'])
    mvn2 = MultivariateGaussGenerator(['x1', 'x2', 'x3'], cov=[[1, 0.5, 0.1], [0.5, 1, 0.05], [0.1, 0.05, 1]])
    ig1 = IndependentGenerator(['x1', 'x2'], [stats.uniform(loc=1, scale=2), stats.uniform(loc=-1, scale=3)])
    ig2 = IndependentGenerator('x1', stats.norm())
    
    smp1, df_smp1 = mvn1.rvs(100)
    smp2, df_smp2 = mvn2.rvs(100)
    smp3, df_smp3 = ig1.rvs(100)
    smp4, df_smp4 = ig2.rvs(100)
    