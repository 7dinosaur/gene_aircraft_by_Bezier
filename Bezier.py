import numpy as np
from math import comb
from scipy import interpolate as si
from matplotlib import pyplot as plt

def Bezier_point(P, n:int=100):
    t = np.linspace(0, 1, n) ##默认100个点
    P = np.array(P, dtype=np.float64)
    assert (P.shape[0] - 1)%3 == 0, "控制点数量必须是3n+1(三阶贝塞尔)"

    weight = np.array([comb(3, i)*((1-t)**(3-i))*t**i for i in range(4)]).T  ## 三阶贝塞尔曲线系数向量
    coords = weight @ P ## 以防未来的你变成弱智，“@”是矩阵乘法

    return coords

class Bezier:
    def __init__(self, P: list) -> None:
        leading_P = P[0]



def draw_curve(ctrl_p):
    #根据控制点生成分段曲线
    t = np.linspace(0, 1, 100)
    slices = [slice(0,4), slice(3,7), slice(6,10)]
    curve = np.zeros([ctrl_p.shape[0], 100, 3])
    for idx, arr in enumerate(ctrl_p):
        local_curve = np.array([Bezier_point(t, arr[s]) for s in slices])

        #对贝塞尔曲线关于x等距插值
        x = np.linspace(0.0, arr[-1, 0], 100)
        tmp_curve = np.zeros([100, 3])
        tmp_curve[:, 0] = x
        local_curve = np.unique(np.concatenate([local_curve[0], local_curve[1], local_curve[2]]), axis=0)
        tmp_curve[:, 1] = si.interp1d(local_curve[:, 0], local_curve[:, 1], kind=2)(x)
        tmp_curve[:, 2] = si.interp1d(local_curve[:, 0], local_curve[:, 2], kind=2)(x)
        curve[idx] = tmp_curve.copy()

    return curve

if __name__ == "__main__":
    P = np.loadtxt("ctrl_P.dat", encoding="utf-8")
    coord = Bezier_point(P.reshape([4, 3]))
    plt.plot(coord[:, 0], coord[:, 2])
    plt.show()