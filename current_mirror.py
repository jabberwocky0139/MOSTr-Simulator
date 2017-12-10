from device import pMOS
from device import Resistor
from device import FreeWire
import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
from mpl_toolkits.mplot3d import Axes3D


class CurrentMirror(object):
    def __init__(self, device='Resistor', unit=1, name='R', param='1'):
        self.Vdd, self.GND = 5.0, 0
        # self.Vin_arr = np.linspace(0, self.Vdd, 1000)

        self.pmos1 = pMOS(unit=1)
        self.pmos2 = pMOS(unit=2)
        self.R1 = Resistor(unit=1, R=10.0)
        self.R2 = Resistor(unit=2, R=5.0)
        self.wire1 = FreeWire()
        self.wire2 = FreeWire()

        if device not in ('nMOS', 'pMOS', 'Resistor'):
            raise NameError('mos: Invalid name')

        self.joint_wire()
        self.process()

    def joint_wire(self):
        # FixedWireとの接続
        self.pmos1.Vs = self.Vdd
        self.pmos2.Vs = self.Vdd
        self.R1.Vl = self.GND
        self.R2.Vl = self.GND

        self.wire1.joint('Drain', self.pmos1)
        self.wire1.joint('Gate', self.pmos1)
        self.wire1.joint('Gate', self.pmos2)
        self.wire1.joint('ResistHi', self.R1)

        self.wire2.joint('Drain', self.pmos2)
        self.wire2.joint('ResistHi', self.R2)

    def process(self):
        R2_arr = np.linspace(1, self.R1.R * 2, 100)
        V1_arr = []  # 動かすパラメータごとに保持
        I1_arr = []  # 動かすパラメータごとに保持
        V2_arr = []  # 動かすパラメータごとに保持
        I2_arr = []  # 動かすパラメータごとに保持
        for R2 in R2_arr:
            self.R2.R = R2
            self.wire1.optimisation()
            self.wire2.optimisation()
            V1_arr.append(self.wire1.voltage)
            I1_arr.append(self.wire1.current)
            V2_arr.append(self.wire2.voltage)
            I2_arr.append(self.wire2.current)
            print('R2: {0}\t region: {1}'.format(self.R2.R, self.pmos2.region))

        plt.plot(R2_arr, V1_arr, label=r'$V_1$')
        plt.plot(R2_arr, V2_arr, label=r'$V_2$')
        plt.xlabel(r'$R_2\ $(R1 = {0})'.format(self.R1.R), fontsize=18)
        plt.ylabel(r'$Voltage$', fontsize=18)
        plt.legend()
        plt.show()
        plt.plot(R2_arr, I1_arr, label=r'$I_1$')
        plt.plot(R2_arr, I2_arr, label=r'$I_2$')
        plt.xlabel(r'$R_2\ $(R1 = {0})'.format(self.R1.R), fontsize=18)
        plt.ylabel(r'$Current$', fontsize=18)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    CurrentMirror()
