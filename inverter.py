from device import nMOS
from device import pMOS
from device import FreeWire
import matplotlib.pyplot as plt
import numpy as np


class Inverter(object):
    def __init__(self, mos, name, param):
        # 引数は動かすパラメータとそのMOS
        # --*-- unitで分けなければならない --*--
        # --*-- 抵抗でもいいでしょ？ --*--
        # --*-- たぶん親クラス作ったほうがいい --*--
        self.Vdd, self.GND = 3, 0
        self.Vin_arr = np.linspace(0, 3, 1000)
        self.Vout_arr = []  # 動かすパラメータごとに保持
        self.Ileak_arr = []  # 動かすパラメータごとに保持

        self.nmos1 = nMOS(unit=1)
        self.pmos1 = pMOS(unit=1)
        self.wire1 = FreeWire()
        if mos not in ('nMOS', 'pMOS'):
            raise NameError('mos: Invalid name')

        self.joint_wire()
        self.process(mos=mos, name=name, param=param)

    def joint_wire(self):
        # FixedWireとの接続
        self.nmos1.Vs = self.GND
        self.pmos1.Vs = self.Vdd
        self.wire1.joint('Drain', self.nmos1)
        self.wire1.joint('Drain', self.pmos1)

    def process(self, mos='nMOS', name='L', param=1):
        mos_obj = self.nmos1 if mos == 'nMOS' else self.pmos1
        setattr(mos_obj, name, param)  # mos_obj.__dict__[name] = param
        change_mos = mos_obj.__class__.__name__

        for Vin in self.Vin_arr:
            self.nmos1.Vg = Vin
            self.pmos1.Vg = Vin
            self.wire1.optimisation()
            self.Vout_arr.append(self.wire1.voltage)
            self.Ileak_arr.append(self.wire1.current)

        plt.plot(self.Vin_arr, self.Vout_arr,
                 label='Vout: {0} {1}={2}'.format(change_mos, name, param))
        plt.plot(self.Vin_arr, self.Ileak_arr,
                 label='Ileak: {0} {1}={2}'.format(change_mos, name, param))


def inverter():
    # Inverter
    for L in (0.5, 1, 1.5, 2.0, 10):
        Inverter('pMOS', 'L', L)

    plt.xlabel(r'$V_{in}$', fontsize=18)
    plt.ylabel(r'$V_{out}\ /\ I_{leak}$', fontsize=18)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    inverter()
