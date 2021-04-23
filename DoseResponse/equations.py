""" Basic Equations """

# Python Dependencies
import numpy as np


# Functions
class Equations(object):
    """A Container Class for Various Curve Fitting Equations."""

    def __init__(self):
        pass

    # Dose Response Fits
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
    def BellShapedDoseResponse(x, Bottom, Top_1, Top_2, pEC50_1, pEC50_2, HillSlope_1, HillSlope_2):
        section_1 = (Top_1 - Bottom) / (1 + 10 ** ((pEC50_1 - x) * HillSlope_1))
        section_2 = (Top_2 - Bottom) / (1 + 10 ** ((x - pEC50_2) * HillSlope_2))
        return Bottom + section_2 + section_1

    # Binding Equations
    @staticmethod
    def OneSiteTotalBinding(x, Background, NS, Bmax, Kd):
        return Bmax * x / (Kd + x) + NS * x + Background

    @staticmethod
    def OneSiteSpecificBinding(x, Bmax, Kd):
        return Bmax * x / (Kd + x)

    @staticmethod
    def SlopedSpecificBinding(x, Bmax, Kd, HillSlope):
        return (Bmax * (x ** HillSlope)) / (Kd ** HillSlope + x ** HillSlope)

    # Kinetic Fits
    @staticmethod
    def PadeApproximant(x, A0, A1, B1):
        return (A0 + A1 * x) / (1 + B1 * x)

    @staticmethod
    def DissociationKinetics(x, Y0, NS, K):
        return (Y0 - NS) * np.exp(-K * x) + NS

    @staticmethod
    def AssociationKinetics(x, Kon, Koff, ligand, Bmax):
        L = ligand * 10 ** (-9)
        Kobs = Kon * L + Koff
        occupancy = L / (L + Koff / Kon)
        return occupancy * Bmax * (1 - np.exp(-Kobs * x))

    @staticmethod
    def TwoPhaseExponentialDecay(x, Y0, Bottom, Percent, Kfast, Kslow):
        fast = (Y0 - Bottom) * Percent
        slow = (Y0 - Bottom) * (1 - Percent)
        return Bottom + fast * np.exp(-Kfast * x) + slow * np.exp(-Kslow * x)
