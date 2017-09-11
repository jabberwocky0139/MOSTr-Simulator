# - 今後の予定 -

# ・負の Vgs/Vds にも対応
#   ▶ とりあえず、Vs は Vd よりも大きくならないという仮定を入れることにする(nMOSの場合)
#     FreeWireにもその仮定を実装することにする

# ・FreeWireの実装．最適化ソルバー
#   ▶ 完了

# ・FreeWireとMOSの連携
#   ▶ 現状mainにぶん投げ. Interfaceクラスを作る

# ・FreeWire: generate_constraintsの改善
#   ▶ 不要

# ・MOS: __call__でなく@propertyで書き換える
#   ▶ 完了

# ・FreeWire: Gate接続時の振る舞い

import matplotlib.pyplot as plt
import math as m
import numpy as np
from scipy import optimize
from abc import ABCMeta, abstractmethod


class VgdsProperty(metaclass=ABCMeta):
    # Vg/Vd/Vsに関するpropertyを定義
    # 代入が行われたときにVgs/Vdsを再計算する
    def __init__(self):
        self._Vg, self._Vd, self._Vs, self._Vgs, self._Vds = [0] * 5

    def calc(self):
        # pMOSのことも考慮して絶対値を取る
        self._Vgs = abs(self._Vg - self._Vs)
        self._Vds = abs(self._Vd - self._Vs)

    @property
    def Vg(self):
        return self._Vg

    @Vg.setter
    def Vg(self, Vg):
        self._Vg = Vg
        self.calc()

    @property
    def Vd(self):
        return self._Vd

    @Vd.setter
    def Vd(self, Vd):
        self._Vd = Vd
        self.calc()

    @property
    def Vs(self):
        return self._Vs

    @Vs.setter
    def Vs(self, Vs):
        self._Vs = Vs
        self.calc()

    @property
    def Vgs(self):
        # Vgs/Vdsはgetterのみ定義(setterに例外入れてもいいんだけど、、、)
        return self._Vgs

    @property
    def Vds(self):
        # Vgs/Vdsはgetterのみ定義
        return self._Vds


class MOS(VgdsProperty):
    # 動作中にDとSが反転するようなことはないと仮定(Vd > Vs)
    def __init__(self, unit='unknown', L=1, W=1, mu=1, Cox=1, lmd=0, Vth=0.5):
        super().__init__()
        self.L, self.W = L, W  # チャネル長 / 幅
        self.mu = mu  # キャリアの移動度
        self.Cox = Cox  # ゲート酸化膜容量
        self.lmd = lmd  # チャネル長変調．線形-非線形領域の接続を考慮して、とりあえず無視
        self.Vth = Vth  # しきい値
        self.beta = self.W / self.L * self.mu * self.Cox
        self.Id = 0  # 正ならD→S方向, 負ならS→D方向
        self.unit = unit  # インスタンス名

    def __call__(self, name, value):
        # --*-- 混乱を招く恐れがあるので未使用 --*--
        # 電圧設定. インスタンスを関数っぽく呼ぶと各ゲートの電圧設定が可
        # Vg/Vd/Vs を設定すると、同時に Vds/Vgs を再設定
        # pMOSでもVgs/dsを正で保持
        if name in ('Vg', 'Vd', 'Vs'):
            self.__dict__['_' + name] = value
            self.__dict__['Vds'] = abs(self._Vs - self._Vd)
            self.__dict__['Vgs'] = abs(self._Vs - self._Vg)
        else:
            raise NameError('Invalid name!')

    @abstractmethod
    def exception(self):
        # Vdsの条件
        pass

    def non_linear(self):
        self.Id = self.beta * ((self.Vgs - self.Vth) - 0.5 * self.Vds) * self.Vds

    def linear(self):
        self.Id = 0.5 * self.beta * (self.Vgs - self.Vth)**2 * (1 + self.lmd * self.Vds)

    def weak_inversion(self):
        self.Id = 0

    def condition(self):
        self.exception()  # 負のVdsに対応していないため、例外を噛ませる

        if self.Vgs < 0:
            return 0
        elif self.Vgs < self.Vth:
            return self.weak_inversion()
        elif self.Vds < self.Vgs - self.Vth:
            return self.non_linear()
        else:
            return self.linear()


class nMOS(MOS):
    def __init__(self, unit='unknown', L=1, W=1, mu=1, Cox=1, lmd=0, Vth=0.5):
        super().__init__(unit, L, W, mu, Cox, lmd, Vth)
        self.ID = 'nMOS No.' + str(unit)

    def exception(self):
        # Vds/Vgsが負になってはいけない例外
        if self.Vds < 0:
            raise ValueError('{0}: Vds is negative!'.format(self.ID))


class pMOS(MOS):
    def __init__(self, unit='unknown', L=1, W=1, mu=0.5, Cox=1, lmd=0, Vth=0.5):
        # nMOSに比べて移動度半分くらい. 他はてきとう
        super().__init__(unit, L, W, mu, Cox, lmd, Vth)
        self.ID = 'pMOS No.' + str(unit)

    def exception(self):
        # Vds/Vgsが正になってはいけない例外
        # pMOSでもVgs/dsを正で保持するので、条件としてはnMOSと同様
        if self.Vds < 0:
            raise ValueError('{0}: Vds is positive!'.format(self.ID))


class FreeWire(object):
    def __init__(self):
        # Gate/Source/Drain端子に接続しているMOSのインスタンスを保持
        self.Drain = []
        self.Source = []
        self.Gate = []

        self.voltage = None  # FreeWireの電圧
        self.current = None  # FreeWireの電流

    def __call__(self, name, instance):
        # Gate/Source/Drain端子と接続
        self.__dict__[name].append(instance)

    def current_law(self, voltage):
        # FreeWireがそれぞれどこに接続されたかによって出力値を変える
        # FreeWireがGateに接続された場合は未実装
        result = 0
        self.current = 0
        for drain in self.Drain:
            drain.Vd = voltage  # Vdをvoltage(二分法のパラメータ)に設定
            drain.condition()  # Idの算出
            if drain.__class__.__name__ is 'nMOS':
                result -= drain.Id  # nMOSのDrainと接続されてたら流出
            else:
                result += drain.Id  # pMOSのDrainと接続されてたら流入
                self.current += drain.Id  # 貫通電流を流入で換算

        for source in self.Source:
            source.Vs = voltage  # Vsをvoltage(二分法のパラメータ)に設定
            source.condition()  # Idの算出
            if source.__class__.__name__ is 'nMOS':
                result += source.Id  # nMOSのSoueceと接続されてたら流入
                self.current += drain.Id  # 貫通電流を流入で換算
            else:
                result -= source.Id  # pMOSのSoueceと接続されてたら流出

        return result

    def generate_constraints(self):
        # nMOSだとVd > Vs, pMOSだとVs > Vdを守れるような範囲を返す
        min_arr, max_arr = [], []

        for drain in self.Drain:
            if drain.__class__.__name__ is 'nMOS':
                min_arr.append(drain.Vs)
            else:
                max_arr.append(drain.Vs)

        for source in self.Source:
            if source.__class__.__name__ is 'nMOS':
                max_arr.append(source.Vd)
            else:
                min_arr.append(source.Vd)

        return max(min_arr), min(max_arr)

    def optimisation(self):
        # brent法で最適化
        # 二分法の上位互換なので、端点[a, b]が解になった場合は検出できない
        # ▶ 端点のみ条件分岐で対応
        a, b = self.generate_constraints()

        if abs(self.current_law(a)) < 1e-6:
            self.voltage = a
        elif abs(self.current_law(b)) < 1e-6:
            self.voltage = b
        else:
            self.voltage = optimize.brentq(self.current_law, a, b)


# FixedWire
Vdd, GND = 3, 0
output_arr = []
output_i_arr = []
Vin_arr = np.linspace(0, 3, 1000)
for L in [0.5, 1, 1.5, 2.0, 10]:
    for Vin in Vin_arr:
        # 初期条件
        nmos1 = nMOS(unit=1)
        # nmos1('Vs', GND)
        # nmos1('Vg', Vin)
        nmos1.Vs = GND
        nmos1.Vg = Vin

        pmos1 = pMOS(unit=1, L=L)
        # pmos1('Vs', VDD)
        # pmos1('Vg', Vin)
        pmos1.Vs = Vdd
        pmos1.Vg = Vin

        # Wire接続
        wire = FreeWire()
        wire('Drain', nmos1)
        wire('Drain', pmos1)

        wire.optimisation()
        output_arr.append(wire.voltage)
        output_i_arr.append(wire.current)

    plt.plot(Vin_arr, output_arr, label='Vout: L={0}'.format(L))
    plt.plot(Vin_arr, output_i_arr, label='Ileak: L={0}'.format(L))
    output_arr = []
    output_i_arr = []

plt.xlabel(r'$V_{in}$', fontsize=18)
plt.ylabel(r'$V_{out}\ /\ I_{leak}$', fontsize=18)
plt.legend()
plt.show()
