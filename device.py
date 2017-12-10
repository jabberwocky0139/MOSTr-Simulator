# - 今後の予定 -

# ・負の Vgs/Vds にも対応
#   ▶ とりあえず、Vs は Vd よりも大きくならないという仮定を入れることにする(nMOSの場合)
#     FreeWireにもその仮定を実装することにする
#   ▶ 完了

# ・FreeWireの実装．最適化ソルバー
#   ▶ 完了

# ・FreeWireとMOSの連携
#   ▶ 現状mainにぶん投げ. Interfaceクラスを作る
#   ▶ 完了

# ・FreeWire: generate_constraintsの改善
#   ▶ 不要

# ・MOS: __call__でなく@propertyで書き換える
#   ▶ 完了

# ・FreeWire: Gate接続時の振る舞い
#   ▶ ゲートリークは無いと仮定．ゲート電流は流れない

# ・FreeWire: もっと簡潔に

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import optimize
from abc import ABCMeta, abstractmethod


class MOS(metaclass=ABCMeta):
    # Vg / Vd / Vs に応じてIdを出力するクラス
    # 動作中にDとSが反転するようなことはないと仮定(Vd > Vs)
    def __init__(self, unit='unknown', L=1, W=1, mu=1, Cox=1, lmd=0.1, Vth=0.7):
        self.Vg, self.Vd, self.Vs = 0, 0, 0
        self._Vgs, self._Vds = 0, 0
        self.L, self.W = L, W  # チャネル長 / 幅
        self.mu = mu  # キャリアの移動度
        self.Cox = Cox  # ゲート酸化膜容量
        self.lmd = lmd  # チャネル長変調．線形-非線形領域の接続を考慮して、とりあえず無視
        self.Vth = Vth  # しきい値
        self._beta = self.W / self.L * self.mu * self.Cox
        self._gm = 0
        self._Id = 0  # 正ならD→S方向, 負ならS→D方向
        self.pcolor = 'navy'  # plotの色を保持
        self.unit = unit  # インスタンス名
        self.region = None

    @property
    def Vgs(self):
        # Vgs参照時に算出
        self._Vgs = abs(self.Vg - self.Vs)
        return self._Vgs

    @property
    def Vds(self):
        # Vds参照時に算出
        self._Vds = abs(self.Vd - self.Vs)
        return self._Vds

    @property
    def beta(self):
        # beta参照時に算出
        self._beta = self.W / self.L * self.mu * self.Cox
        return self._beta

    @property
    def gm(self):
        tmp_Id = self.Id
        self.Vg += 0.01
        self._gm = (self.Id - tmp_Id) / 0.01
        self.Vg -= 0.01
        return self._gm

    @abstractmethod
    def _exception(self):
        # Vdsの条件
        pass

    def _non_linear(self):
        # グラフの色を同時に出力(Blue)
        self._Id = self.beta * ((self.Vgs - self.Vth) - 0.5 * self.Vds) * self.Vds
        self.pcolor, self.region = 'b', 'non-linear'
        return self._Id

    def _linear(self):
        # グラフの色を同時に出力(Green)
        self._Id = 0.5 * self.beta * (self.Vgs - self.Vth)**2 * (1 + self.lmd * self.Vds)
        self.pcolor, self.region = 'g', 'linear'
        return self._Id

    def _weak_inversion(self):
        # グラフの色を同時に出力(Red)
        self._Id = 0
        self.pcolor, self.region = 'r', 'weak-inversion'
        return self._Id

    @property
    def Id(self):
        self._exception()  # 負のVdsに対応していないため、例外を噛ませる

        if self.Vgs < 0:
            return 0
        elif self.Vgs < self.Vth:
            return self._weak_inversion()
        elif self.Vds < self.Vgs - self.Vth:
            return self._non_linear()
        else:
            return self._linear()


class nMOS(MOS):
    def __init__(self, unit='unknown', L=1, W=1, mu=2, Cox=1, lmd=0, Vth=0.7):
        super().__init__(unit, L, W, mu, Cox, lmd, Vth)
        self.ID = 'nMOS No.' + str(self.unit)

    def _exception(self):
        # Vds/Vgsが負になってはいけない例外
        if self.Vd < self.Vs:
            raise ValueError('{0}: Vds is negative!'.format(self.ID))


class pMOS(MOS):
    def __init__(self, unit='unknown', L=1, W=1, mu=0.5, Cox=1, lmd=0, Vth=0.7):
        # nMOSに比べて移動度半分くらい. 他はてきとう
        super().__init__(unit, L, W, mu, Cox, lmd, Vth)
        self.ID = 'pMOS No.' + str(self.unit)

    def _exception(self):
        # Vds/Vgsが正になってはいけない例外
        # pMOSでもVgs/dsを正で保持するので、条件としてはnMOSと同様
        if self.Vs < self.Vd:
            raise ValueError('{0}: Vds is positive!'.format(self.ID))


class FreeWire(object):
    # 素子と接続して、キルヒホッフの電流則を守るように電圧を決定する
    def __init__(self, unit='unknown'):
        self.unit = str(unit)
        # Gate/Source/Drain端子に接続しているMOSのインスタンスを保持
        self.Drain = []  # MOSのDrain
        self.Source = []  # MOSのSource
        self.Gate = []  # MOSのGate ゲートリーク無し
        self.ResistHi = []  # 抵抗のHi端子
        self.ResistLo = []  # 抵抗のLo端子

        self.voltage = 0  # FreeWireの電圧
        self.current = 0  # FreeWireの電流
        self.previous_voltage = -np.inf
        self.previous_current = -np.inf

    def __call__(self, name, instance):
        # Gate/Source/Drain端子と接続
        # jointのシンタックスシュガー
        self.joint(name, instance)

    def joint(self, name, instance):
        # Gate/Source/Drain端子と接続
        self.__dict__[name].append(instance)

    def current_law(self, voltage):
        # FreeWireがそれぞれどこに接続されたかによって出力値を変える
        # FreeWireがGateに接続された場合は未実装
        # ゲートリークなどな無いと仮定して、ゲートには電流が流れないものとする
        # なんかもっと上手く書けないかな、、、

        result = 0
        self.current = 0

        # MOSとのGate接続
        for gate in self.Gate:
            gate.Vg = voltage  # Gateには電流は流れないので、電圧設定だけ

        # MOSとのDrain接続
        for drain in self.Drain:
            drain.Vd = voltage  # Vdをvoltage(二分法のパラメータ)に設定
            if drain.__class__.__name__ is 'nMOS':
                result -= drain.Id  # nMOSのDrainと接続されてたら流出
            else:
                result += drain.Id  # pMOSのDrainと接続されてたら流入
                self.current += drain.Id  # FreeWireの電流(貫通電流)を流入で換算

        # MOSとのSource接続
        for source in self.Source:
            source.Vs = voltage  # Vsをvoltage(二分法のパラメータ)に設定
            if source.__class__.__name__ is 'nMOS':
                result += source.Id  # nMOSのSoueceと接続されてたら流入
                self.current += drain.Id  # FreeWireの電流(貫通電流)を流入で換算
            else:
                result -= source.Id  # pMOSのSoueceと接続されてたら流出

        # 抵抗とのHi側接続
        for Rh in self.ResistHi:
            Rh.Vh = voltage
            result -= Rh.Ir  # Hi側との接続なら流出

        # 抵抗とのLo側接続
        for Rl in self.ResistLo:
            Rl.Vl = voltage
            result += Rl.Ir  # Hi側との接続なら流出
            self.current += Rl.Ir  # FreeWireの電流(貫通電流)を流入で換算

        return result

    def generate_constraints(self):
        # nMOSだとVd > Vs, pMOSだとVs > Vd, 抵抗だと Vh > Vlを守れるような範囲を返す
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

        for Rh in self.ResistHi:
            min_arr.append(Rh.Vl)

        for Rl in self.ResistLo:
            max_arr.append(Rl.Vh)

        # print(min_arr, max_arr)  # debug

        return max(min_arr), min(max_arr)

    def optimisation(self):
        # brent法で最適化
        # 二分法の上位互換なので、端点[a, b]が解になった場合は検出できない
        # ▶ 端点のみ条件分岐で対応
        # previous_voltage との差分が1e-6以下ならFalse, それ以外ならTrueを返す
        a, b = self.generate_constraints()

        # print(a, b)  # debug

        if abs(self.current_law(a)) < 1e-6:
            self.voltage = a
        elif abs(self.current_law(b)) < 1e-6:
            self.voltage = b
        else:
            self.voltage = optimize.brentq(self.current_law, a, b)

        flag = abs(self.voltage - self.previous_voltage) > 1e-6
        self.previous_voltage = self.voltage
        self.previous_current = self.current

        return flag


class Capacitor(object):
    # 端子に電圧がかかった時間に応じて電荷を蓄積する
    # 内部電荷で電圧が確定
    # 片方の電位をFixしてもう片方の電位を電荷で決めるイメージ
    # 両端子にFreeWireが接続されていたら、、、
    # 「時間」の概念が導入されるので、FreeWireの最適化問題と組み合わせると
    # 計算時間が爆発しそう．厄介なのでとりあえず保留
    # コンパレートとAZは切り分けて考えたほうがよさそう
    def __init__(self, unit='unknown'):
        self.C = 1
        self.Q = 0
        self.Va, self.Vb = 0, 0


class Resistor(object):
    # 端子電圧によって電流を決定する
    # 電流はVh→Vlに流れるのが正とする．FreeWireの実装もこれに合わせて
    def __init__(self, unit='unknown', R=1):
        self.R = R
        self.Vh, self.Vl = 0, 0
        self._Ir = 0  # 正ならD→S方向, 負ならS→D方向
        self.unit = unit

    @property
    def Ir(self):
        self._Ir = (self.Vh - self.Vl) / self.R
        return self._Ir


# --*-- 関数エリア --*--

def plot_mos_gm():
    nmos1 = nMOS(unit=1)
    Vdd = 5

    for Vd in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        nmos1.Vs, nmos1.Vd = 0, Vd
        Vg_arr = np.linspace(0, Vdd, 100)
        Id_arr = []

        for Vg in Vg_arr:
            nmos1.Vg = Vg
            Id_arr.append(nmos1.Id)

        plt.plot(Vg_arr, np.gradient(Id_arr, Vdd/100), label='Vds={0}'.format(Vd))

    plt.xlabel(r'$V_{gs}$', fontsize=18)
    plt.ylabel(r'$g_m$', fontsize=18)
    plt.legend()
    plt.show()


def plot_mos_3d():
    # 3Dplot
    nmos1 = nMOS(unit=1)
    vg = np.linspace(0, 3.0, 100)
    vd = np.linspace(0, 3.0, 100)
    Vg, Vd = np.meshgrid(vg, vd)
    Id_2darr = [[0 for _ in range(100)] for __ in range(100)]
    fig = plt.figure()
    ax = Axes3D(fig)

    for i in range(100):
        for j in range(100):
            nmos1.Vg = Vg[i][j]
            nmos1.Vd = Vd[i][j]
            Id_2darr[i][j]= nmos1.Id
            # ax.scatter3D(nmos1.Vg, nmos1.Vd, Id_2darr[i][j], c=nmos1.pcolor)

    ax.plot_wireframe(Vg, Vd, Id_2darr)
    ax.set_xlabel(r'$V_{gs}$', fontsize=18)
    ax.set_ylabel(r'$V_{ds}$', fontsize=18)
    ax.set_zlabel(r'$I_d$', fontsize=18)
    plt.legend()
    plt.show()


def plot_mos_operate_point():
    nmos1 = nMOS(unit=1)
    Vdd = 5

    for Vg in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        nmos1.Vs, nmos1.Vg = 0, Vg
        Vd_arr = np.linspace(0, Vdd, 100)
        Id_arr = []

        for Vd in Vd_arr:
            nmos1.Vd = Vd
            Id_arr.append(nmos1.Id)

        plt.plot(Vd_arr, Id_arr, label='Vgs={0}'.format(Vg))

    for R in [1, 3, 6, 9, 12, 15]:
        operate_arr = [(Vdd - Vds) / R for Vds in Vd_arr]
        plt.plot(Vd_arr, operate_arr, label='operate R={0}'.format(R))

    plt.xlabel(r'$V_{ds}$', fontsize=18)
    plt.ylabel(r'$I_d$', fontsize=18)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_mos_gm()
    pass
