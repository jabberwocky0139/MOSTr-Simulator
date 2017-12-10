from device import nMOS
from device import Resistor
from device import FreeWire
import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
from mpl_toolkits.mplot3d import Axes3D


class CommonSource(metaclass=ABCMeta):
    def __init__(self, element='Resistor', unit=1, name='R', param='1'):
        self.Vdd, self.GND = 5.0, 0
        self.time = np.linspace(0, 6 * np.pi, 1000)
        self.Vin_DC = 0.74
        self.Vin_arr = np.linspace(0, self.Vdd, 1000)
        self.Vin_time_arr = 0.01 * np.sin(self.time) + self.Vin_DC
        self.Vout_arr = []  # 動かすパラメータごとに保持
        self.Ileak_arr = []  # 動かすパラメータごとに保持

        self.nmos1 = nMOS(unit=1, mu=1)
        self.R1 = Resistor(unit=1, R=2500.0)
        self.wire1 = FreeWire()
        if element not in ('nMOS', 'pMOS', 'Resistor'):
            raise NameError('mos: Invalid name')

        self.joint_wire()
        self.process()

    def joint_wire(self):
        # FixedWireとの接続
        self.nmos1.Vs = self.GND
        self.R1.Vh = self.Vdd

        self.wire1.joint('Drain', self.nmos1)
        self.wire1.joint('ResistLo', self.R1)

    @abstractmethod
    def process(self):
        pass


class CSTimeVout(CommonSource):
    def __init__(self, element='Resistor', unit=1, name='R', param='1'):
        super().__init__(element, unit, name, param)

    def process(self):
        for Vin in self.Vin_time_arr:
            self.nmos1.Vg = Vin
            self.wire1.optimisation()
            self.Vout_arr.append(self.wire1.voltage)
            self.Ileak_arr.append(self.wire1.current)
            print('Vin(Vgs) = {0:1.3f}, Vout(Vds) = {2:1.3f}, gm = {1:1.3f}, Rd={4:1.3f}, gmRd = {3:1.3f}'.format(
                Vin, self.nmos1.gm, self.nmos1.Vds, self.nmos1.gm * self.R1.R, self.R1.R))

        Av = (max(self.Vout_arr) - min(self.Vout_arr))/(max(self.Vin_time_arr) - min(self.Vin_time_arr))
        Vout = np.array(self.Vout_arr) + self.Vin_DC - self.Vout_arr[0]
        genuin_Vout = -0.01 * np.sin(self.time) * Av + self.Vin_DC
        diff = max(Vout) - max(genuin_Vout)

        print('Av = {0:1.3f}'.format(-Av))
        plt.plot(self.time, self.Vin_time_arr, label='Vin')
        plt.plot(self.time, Vout, label='Vout')
        plt.plot(self.time, genuin_Vout + diff, label='genuin amplifier')


class CSVinVout(CommonSource):
    def __init__(self, element='Resistor', unit=1, name='R', param='1'):
        super().__init__(element, unit, name, param)

    def process(self):
        for Vin in self.Vin_arr:
            self.nmos1.Vg = Vin
            self.wire1.optimisation()
            self.Vout_arr.append(self.wire1.voltage)
            self.Ileak_arr.append(self.wire1.current)

            # plt.plot(Vin, self.Vout_arr[-1], '.', color=self.nmos1.pcolor)

        Vout_gradient = np.gradient(self.Vout_arr, self.Vin_arr[1] - self.Vin_arr[0])

        plt.figure(figsize=(3, 2))
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        ax1.plot(self.Vin_arr, self.Vout_arr, label='Vout')
        ax2.plot(self.Vin_arr, Vout_gradient)  # 勾配の計算
        ax1.plot([self.Vin_DC, self.Vin_DC], [0, 5], '-')
        ax2.plot([self.Vin_DC, self.Vin_DC], [min(Vout_gradient), 3], '-')
        ax2.plot([self.Vin_DC-0.01, self.Vin_DC-0.01], [min(Vout_gradient), 3], 'k-')
        ax2.plot([self.Vin_DC+0.01, self.Vin_DC+0.01], [min(Vout_gradient), 3], 'k-')
        ax2.set_xlabel(r'$V_{in}$', fontsize=18)
        ax1.set_ylabel(r'$V_{out}$', fontsize=18)
        ax2.set_ylabel(r'$A_v$', fontsize=18)
        ax1.set_xlim(0.6, 1.0)
        ax2.set_xlim(0.6, 1.0)
        ax2.set_ylim(min(Vout_gradient)-2, 5)


class CSVinRdgmRd(CommonSource):
    def __init__(self, element='Resistor', unit=1, name='R', param='1'):
        super().__init__(element, unit, name, param)

    def process(self):
        xdim, ydim = 200, 200
        vg = np.linspace(0, self.Vdd-2, xdim)
        rd = np.linspace(1, 2000, ydim)
        Vg, Rd = np.meshgrid(vg, rd)
        gmRd_arr = [[0 for _ in range(ydim)] for __ in range(xdim)]
        fig = plt.figure()
        ax = Axes3D(fig)

        for i in range(xdim):
            for j in range(ydim):
                self.nmos1.Vg = Vg[i][j]
                self.wire1.optimisation()
                self.R1.R = Rd[i][j]
                gmRd_arr[i][j] = self.nmos1.gm * self.R1.R

        ax.plot_wireframe(Vg, Rd, gmRd_arr)
        ax.set_xlabel(r'$V_{gs}$', fontsize=18)
        ax.set_ylabel(r'$R_{d}$', fontsize=18)
        ax.set_zlabel(r'$g_mRd$', fontsize=18)


def common_source_time_Vout():
    CSTimeVout()
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$V_{in}\ /\ V_{out}$', fontsize=18)
    plt.legend()
    plt.show()


def common_source_Vin_Vout():
    CSVinVout()
    plt.legend()
    plt.show()


def common_source_Vin_Rd_gmRd():
    CSVinRdgmRd()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    common_source_time_Vout()
    common_source_Vin_Vout()
