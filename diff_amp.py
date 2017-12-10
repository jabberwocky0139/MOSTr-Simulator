from device import Resistor
from device import nMOS
from device import FreeWire
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum


class DiffAmp(object):
    def __init__(self, device='Resistor', unit=1, name='R', param='1'):
        self.Vdd, self.GND = 5.0, 0

        self.R1 = Resistor(unit=1, R=10)
        self.R2 = Resistor(unit=1, R=10)
        self.nmos1 = nMOS(unit=1)
        self.nmos2 = nMOS(unit=2)
        self.nmos3 = nMOS(unit=3, W=2, Vth=0.5)
        self.mosz = [self.nmos1, self.nmos2, self.nmos3]  # 参照代入

        self.wire1 = FreeWire(unit=1)
        self.wire2 = FreeWire(unit=2)
        self.wire3 = FreeWire(unit=3)
        self.wires = [self.wire1, self.wire2, self.wire3]  # 参照代入

        if device not in ('nMOS', 'pMOS', 'Resistor'):
            raise NameError('mos: Invalid name')

        self.result = {}
        for wire in self.wires:
            self.result['V'+wire.unit] = []
            self.result['I'+wire.unit] = []

        self.process()

    def joint_wire(self):
        # FixedWireとの接続
        self.R1.Vh = self.Vdd
        self.R2.Vh = self.Vdd
        self.nmos3.Vs = self.GND
        self.nmos3.Vg = 1.0  # 雑設定

        self.wire1.joint('ResistLo', self.R1)
        self.wire1.joint('Drain', self.nmos1)
        self.wire2.joint('ResistLo', self.R2)
        self.wire2.joint('Drain', self.nmos2)
        self.wire3.joint('Source', self.nmos1)
        self.wire3.joint('Source', self.nmos2)
        self.wire3.joint('Drain', self.nmos3)

    def process(self):

        Vin_DC = 2.0
        time_arr = np.linspace(0, 6 * np.pi, 1000)
        Vin1_time_arr = 0.01 * np.sin(time_arr) + Vin_DC
        Vin2_time_arr = -0.01 * np.sin(time_arr) + Vin_DC

        for time, Vin1, Vin2 in zip(time_arr, Vin1_time_arr, Vin2_time_arr):
            # print('\r', '{0:.2f}%'.format(time / max(time_arr) * 100),
            #       end='', flush=True)
            self.joint_wire()
            self.nmos1.Vg = Vin1
            self.nmos2.Vg = Vin2

            flags = [True] * 3
            while(any(flags)):
                for i, wire in enumerate(self.wires):
                    flags[i] = wire.optimisation()

            for wire in self.wires:
                self.result['V'+wire.unit].append(wire.voltage)
                self.result['I'+wire.unit].append(wire.current)

            for wire in self.wires:
                wire.__init__(unit=wire.unit)
        else:
            for mos in self.mosz:
                print(mos.unit, mos.region)  # for debug

        plt.plot(time_arr, np.array(self.result['V1']) - np.array(self.result['V2']),label=r'$V_1-V_2$')
        A = np.sqrt(2 * self.result['I3'][0]) * self.R1.R
        plt.plot(time_arr, (np.array(Vin1_time_arr) - np.array(Vin2_time_arr)) * A, label=r'$V_{in1}$')
        plt.xlabel(r'$time$', fontsize=18)
        plt.ylabel(r'$Voltage$', fontsize=18)
        plt.legend()
        plt.show()

        print(np.ptp(self.result['I3']))
        # plt.plot(time_arr, self.result['V3'], label=r'$V_3$')
        plt.plot(time_arr, self.result['I3'], label=r'$I_3$')
        plt.xlabel(r'$time$', fontsize=18)
        plt.ylabel(r'$Voltage\ /\ Current$', fontsize=18)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    DiffAmp()
