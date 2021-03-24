""" Basic Equations """

# Python Dependencies
import pandas as pd
import numpy as np


# Functions
class Equations(object):
    """Insert Docstring Here."""
    def __init__(self):
        pass

    @staticmethod
    def BoltzmannSigmoidal(x, Top, Bottom, V50, Slope):
        return Bottom + (Top - Bottom) / (1 + np.e ** ((V50 - x) / Slope))

    @staticmethod
    def FourParameterDoseResponse(x, Top, Bottom, pEC50, HillSlope):
        """x Must be in units of Log for this to work"""
        return Bottom + (Top - Bottom) / (1 + 10 ** ((pEC50 - x) * HillSlope))
    
    @staticmethod
    def VariableSlopeDoseResponse(x, Top, Bottom, EC50, HillSlope):
        """x is not converted to log"""
        return Bottom + (x ** HillSlope) * (Top - Bottom) / (x ** HillSlope + EC50 ** HillSlope)

    @staticmethod
    def AsymmetricSlopeDoseResponse(x, Top, Bottom, EC50, HillSlope, S):
        denomerator = (1 + (2 ** (1/S) - 1) * ((EC50 / x) ** HillSlope)) ** S
        numerator = Top - Bottom
        return Bottom + numerator / denomerator

    @staticmethod
    def BiphasicDoseResponse(x, Top, Bottom, f, EC50_1, EC50_2, HillSlope_1, HillSlope_2):
        span = Top - Bottom
        s_1 = span * f / (1 + (EC50_1 / x) ** HillSlope_1)
        s_2 = span * (1 - f) / (1 + (EC50_2 / x) ** HillSlope_2)
        return Bottom + s_1 + s_2
    
    @staticmethod
    def OneSiteSpecificBinding(x, Bmax, Kd):
        return Bmax * x / (Kd + x)
