# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:03:34 2020

@author: sobol
"""

import numpy as np
from labos_flow_v2 import *
from labos_point import Point

if __name__ == '__main__':
    p1 = Point({'x1' : 1., 'x2' : 1.})
    p2 = Point({'x1' : 2., 'x2' : 3.})
    p3 = Point({'x1' : 0., 'x2' : 0.})
    sample = {'x1' : np.array([1,2,3,4,5,6]).reshape(6,1), 'x2' : np.array([1,2,3,4,5,6]).reshape(6,1)}
    sample2 = {'x1' : np.array([0,1,0,1]).reshape(4,1), 'x2' : np.array([1,0,1,0]).reshape(4,1)}
    test_points = [p1, p2, p3]
    x1 = Identity('x1')
    x2 = Identity('x2')
    f1 = 3.0*x1 + 2.
    f2 = LabSin(x1)
    f3 = x1**2 - 2*x1*x2
    #LabFunc('x1**2 - 2*x1*x2', {'x1': '2*x1 - 2*x2', 'x2' : '-2*x2'})
    for test_point in test_points:
        try:
            assert (f1 + f2)(test_point) == f1(test_point) + f2(test_point)
            assert (f3 + f1)(test_point) == f3(test_point) + f1(test_point)
            assert (f1 + 1)(test_point) == f1(test_point) + 1
            assert (1 + f3)(test_point) == 1 + f3(test_point)
        except Exception as ex:
            print(test_point)
            raise ex
    for test_point in [sample, sample2]:
        try:
            assert ((f1 + f2)(test_point) == f1(test_point) + f2(test_point)).all()
            assert ((f3 + f1)(test_point) == f3(test_point) + f1(test_point)).all()
            assert ((f1 + 1)(test_point) == f1(test_point) + 1).all()
            assert ((1 + f3)(test_point) == 1 + f3(test_point)).all()
        except Exception as ex:
            print(test_point)
            raise ex
    for test_point in test_points:
        # subtract
        assert (f1 - f3)(test_point) == f1(test_point) - f3(test_point)
        assert (f1 - 1)(test_point) == f1(test_point) - 1
        assert (1 - f3)(test_point) == 1 - f3(test_point)
        # multiply
        assert (f1 * f3)(test_point) == f1(test_point)*f3(test_point)
        assert (f1 * 2)(test_point) == 2*f1(test_point)
        assert (3 * f3)(test_point) ==3 * f3(test_point)
        assert (f1 * f2)(test_point) ==f1(test_point) * f2(test_point)
        # divide
        try:
            assert (f1 / f3)(test_point) == f1(test_point)/f3(test_point)
            assert (3/f3)(test_point) == 3/f3(test_point)
        except ZeroDivisionError:
            pass
        assert (f1 / 2)(test_point) == f1(test_point)/2
        # power
        assert (f1 ** f3)(p2) == f1(p2)**f3(p2)
        assert (f1 ** 3)(test_point) == f1(test_point)**3
        assert (3.0**f3)(test_point) == 3.0**f3(test_point)
    for test_point in [sample, sample2]: 
        # subtract
        assert ((f1 - f3)(test_point) == f1(test_point) - f3(test_point)).all()
        assert ((f1 - 1)(test_point) == f1(test_point) - 1).all()
        assert ((1 - f3)(test_point) == 1 - f3(test_point)).all()
        # multiply
        assert ((f1 * f3)(test_point) == f1(test_point)*f3(test_point)).all()
        assert ((f1 * 2)(test_point) == 2*f1(test_point)).all()
        assert ((3 * f3)(test_point) ==3 * f3(test_point)).all()
        assert ((f1 * f2)(test_point) ==f1(test_point) * f2(test_point)).all()
        # divide
        try:
            if (f3(test_point) != 0).all():
                assert ((f1 / f3)(test_point) == f1(test_point)/f3(test_point)).all()
                assert ((3/f3)(test_point) == 3/f3(test_point)).all()
        except ZeroDivisionError:
            pass
        assert ((f1 / 2)(test_point) == f1(test_point)/2).all()
        
        # power
        assert (f1 ** f3)(p2) == f1(p2)**f3(p2)
        assert ((f1 ** 3)(test_point) == f1(test_point)**3).all()
        assert ((3.0**f3)(test_point) == 3.0**f3(test_point)).all()
    """
    now derivatives
    """
    for i, test_point in enumerate(test_points + [sample, sample2]):
        try:
            assert (f1 + f2).deriv(test_point) == f1.deriv(test_point) + f2.deriv(test_point)
            assert (f1 + 1).deriv(test_point) == f1.deriv(test_point)
            assert (1 + f3).deriv(test_point) == f3.deriv(test_point)
            # subtract
            assert (f1 - f3).deriv(test_point) == f1.deriv(test_point) - f3.deriv(test_point)
            assert (f1 - 1).deriv(test_point) == f1.deriv(test_point)
            assert (1 - f3).deriv(test_point) == -f3.deriv(test_point)
            # multiply
            assert (f1 * f3).deriv(test_point) == f1.deriv(test_point)*f3(test_point)  + f3.deriv(test_point)*f1(test_point)
            assert (f1 * 2).deriv(test_point) == 2*f1.deriv(test_point)
            assert (1 * f3).deriv(test_point) == f3.deriv(test_point)
            assert (f1 * f2).deriv(test_point) == f1.deriv(test_point)*f2(test_point)  + f2.deriv(test_point)*f1(test_point)
            assert (f1 * f2 * f3).deriv(test_point) == f1.deriv(test_point)*f2(test_point)*f3(test_point) \
            + f2.deriv(test_point)*f1(test_point)*f3(test_point) + f3.deriv(test_point)*f1(test_point)*f2(test_point)
            # divide
            try:
                if (f3(test_point) != 0).all():
                    assert (f1 / f3).deriv(test_point) == (f1.deriv(test_point)*f3(test_point) - f3.deriv(test_point)*f1(test_point))/(f3(test_point)**2)
                    assert (3/f3).deriv(test_point) == -3*f3.deriv(test_point)/(f3(test_point)**2)
            except ZeroDivisionError:
                pass
            assert (f1 / 2).deriv(test_point) == f1.deriv(test_point)/2
            # power
            assert (f1 ** f3).deriv(p2) == (f3.deriv(p2)*np.log(f1(p2)) + f1.deriv(p2)*f3(p2)/f1(p2)) * (f1(p2)**f3(p2))
            assert (f1 ** 3).deriv(test_point) == 3 * f1.deriv(test_point) * f1(test_point)**2
            assert (3.0**f3).deriv(test_point) == np.log(3) * f3.deriv(test_point) * (3.0**f3(test_point))
        except Exception as ex:
            print(test_point)
            raise ex
    
    
    
    t2 = LabExp(f1)
    t3 = LabLog(t2)
    t4 = LabCos(f2+f3)
    t5 = LabSin(f1+f2)
    
    for point in [p1, p2, p3]:
        assert t2(point) == np.exp(f1(point))
        assert t3(point) == np.log(t2(point))
        assert t4(point) == np.cos(f2(point) + f3(point))
        assert t5(point) == np.sin(f1(point) + f2(point))
        
        assert t2.deriv(point) == f1.deriv(point)*np.exp(f1(point))
        assert t3.deriv(point) == t2.deriv(point)/t2(point)
        assert t4.deriv(point) == -(f2.deriv(point) + f3.deriv(point))*np.sin(f2(point) + f3(point))
        assert t5.deriv(point) == (f1.deriv(point) + f2.deriv(point))*np.cos(f1(point) + f2(point))
        
    for point in [sample, sample2]:
        assert (t2(point) == np.exp(f1(point))).any()
        assert (t3(point) == np.log(t2(point))).any()
        assert (t4(point) == np.cos(f2(point) + f3(point))).any()
        assert (t5(point) == np.sin(f1(point) + f2(point))).any()
        
        assert t2.deriv(point) == f1.deriv(point)*np.exp(f1(point))
        assert t3.deriv(point) == t2.deriv(point)/t2(point)
        assert t4.deriv(point) == -(f2.deriv(point) + f3.deriv(point))*np.sin(f2(point) + f3(point))
        assert t5.deriv(point) == (f1.deriv(point) + f2.deriv(point))*np.cos(f1(point) + f2(point))
    
    
    
     