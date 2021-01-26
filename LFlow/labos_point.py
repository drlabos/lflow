# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:19:14 2020

@author: sobol
"""
import numpy as np

class Point():
    """
    store dictionary with name-value pairs for variables
    """
    def __init__(self, dct):
        self.dict = dct
        
        
    @property
    def _keys(self):
        return set(self.dict.keys())
    
    @property
    def _size(self):
        for key in self.dict.keys():
            return 1 if np.isscalar(self.dict[key]) else self.dict[key].shape[0]
            break
    
    def __getitem__(self, item):
        return self.dict[item]
    
    def __setitem__(self, item, value):
        self.dict[item] = value
    
    
    def __neg__(self):
        return Point({key : -self.dict[key] for key in self.dict})
    
    def __add__(self, other):
        new_dict = self.dict.copy()
        for key in other.dict:
            if key in new_dict:
                new_dict[key] += other[key]
            else:
                new_dict[key] = other[key]
        return Point(new_dict)
    
    def __sub__(self, other):
        new_dict = self.dict.copy()
        for key in other.dict:
            if key in new_dict:
                new_dict[key] -= other[key]
            else:
                new_dict[key] = -other[key]
        return Point(new_dict)
    
    def __eq__(self, other):
        """
        two points are equal only if their keys and values match
        """
        if self._size == 1:
            return self.dict == other.dict
        else:
           return all([(self.dict[k]==other.dict[k]).all() for k in self._keys.union(other._keys)]) 
    
    def __ne__(self, other):
        return self.dict != other.dict
    
    
    def __mul__(self, other):
        """
        support only numbers
        """
        if isinstance(other, Point):
            raise NotImplementedError
        new_dict = self.dict.copy()
        for key in new_dict:
            new_dict[key] *= other
        return Point(new_dict)
    
    
    def __truediv__(self, other):
        """
        support only numbers
        """
        if isinstance(other, Point):
            raise NotImplementedError
        new_dict = self.dict.copy()
        for key in new_dict:
            new_dict[key] = new_dict[key]/other
        return Point(new_dict)
            
            
    def __rmul__(self, other):
        if np.isscalar(other):
            new_dict = self.dict.copy()
            for key in new_dict:
                new_dict[key] *= other
            return Point(new_dict)
        else:
            raise Exception('use left mul! iterable * Point may produce unexpected result')
    
    
    def __str__(self):
        return str(self.dict)
    
    def __repr__(self):
        return str(self)
    
    
    def expand(self, other):
        res = Point(self.dict.copy())
        if isinstance(other, Point):
            res.dict.update(other.dict)
        elif isinstance(other, dict):
            res.dict.update(other)
        return res
    
    def shrink(self, keys):
        return Point({k : self[k] for k in keys})
    
    
    def norm(self):
        return np.sqrt(np.sum([self[k]**2 for k in self.dict]))
    
    
    def cos(self, other):
        return np.sum([self[k]*other[k] for k in self.dict])/(self.norm()*other.norm())
        
        
    