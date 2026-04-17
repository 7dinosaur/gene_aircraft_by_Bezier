import numpy as np
from math import comb
from scipy import interpolate as si
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def Bezier_point(P, n:int=100):
    t = np.linspace(0, 1, n) ##默认100个点
    P = np.array(P, dtype=np.float64)

    weight = np.array([comb(3, i)*((1-t)**(3-i))*t**i for i in range(4)]).T  ## 三阶贝塞尔曲线系数向量
    coords = weight @ P ## 以防未来的你变成弱智，“@”是矩阵乘法

    return coords

def redis(coords, n):
    coords = np.insert(coords, 3, np.zeros(coords.shape[0]), axis=1)
    coords[1:, 3] = np.linalg.norm(np.diff(coords[:, :3], axis=0), axis=1)
    for i in range(1, coords.shape[0]):
        coords[i, 3] += coords[i-1, 3]
    L_axis = np.linspace(0, coords[-1, 3], n)
    L_interfun_x = si.interp1d(coords[:, 3], coords[:, 0], kind='linear')
    L_interfun_y = si.interp1d(coords[:, 3], coords[:, 1], kind='linear')
    L_interfun_z = si.interp1d(coords[:, 3], coords[:, 2], kind='linear')
    new_coords = np.zeros([n, 3])
    new_coords[:, 0] = L_interfun_x(L_axis)
    new_coords[:, 1] = L_interfun_y(L_axis)
    new_coords[:, 2] = L_interfun_z(L_axis)

    return new_coords

def read_x(file):
    with open(file) as f:
        lines = f.readlines()
        n_dom = int(lines[0]) ## 第一行是网格块数
        doms = []
        for i in range(n_dom):
            shape = [int(n) for n in lines[i+1].split()] + [3]
            doms.append(np.zeros(shape)) ## 逐行读取每个网格形状
        points = []
        for line in lines[n_dom+1:]:
            points.extend([float(n) for n in line.split()]) ## 所有坐标点展平
        for idx, dom in enumerate(doms):
            len_dom = dom.shape[0]*dom.shape[1]*dom.shape[2]
            doms[idx] = np.array(points[:len_dom*3]).reshape([3, -1]).T.reshape(dom.shape)
            del(points[:len_dom*3])

    return doms

class Bezier:
    def __init__(self, P) -> None:
        self.leading_P = P[0]
        # self.trailing_P = P[1]
        # self.sec_P = P[2:]

    def write_curve(self, mesh, file_path) -> None:
        with open(file_path, 'w') as f:
            print("写入曲线...")
            f.write(f"{len(mesh)}\n")
            for dom in mesh:
                n_i = dom.shape[0]
                f.write(f"{n_i} 1 1\n")
            for dom in mesh:
                row_size = 4  # 控制每行4个元素
                dom = dom.T
                for coord in dom:
                    coord = coord.flatten()
                    for i in range(0, len(coord), row_size):
                        line_elements = coord[i:i+row_size]
                        line_str = " ".join(f"{x:.6f}" for x in line_elements)
                        f.write(line_str + "\n")
            print(f"写入完毕,面元数为[{len(mesh)}]. 网格文件路径：{file_path}")
    
def draw_curve(ctrl_p):
    #根据控制点生成分段曲线
    assert (ctrl_p.shape[0] - 1)%3 == 0, "控制点数量必须是3n+1(三阶贝塞尔)"
    n_sec = int((ctrl_p.shape[0] - 1)/3)
    P_list = [ctrl_p[3*i:3*(i+1)+1] for i in range(n_sec)]
    curve = []
    for P in P_list:
        curve.append(Bezier_point(P))
    curve = np.concatenate(curve, axis=0)
    print(curve.shape)
    return curve

def loss(x, P_fixed, target_curve):
    # 构建当前贝塞尔
    P = P_fixed.copy()
    P[1] = x[:3]
    P[2] = x[3:6]
    
    bezier_curve = Bezier_point(P)
    
    # ✅ 关键：按弧长对齐两条曲线（真正几何对齐）
    target_aligned, bezier_aligned = redis(target_curve, 100), redis(bezier_curve, 100)
    
    # 计算几何误差
    return np.mean(np.linalg.norm(target_aligned - bezier_aligned, axis=1))

if __name__ == "__main__":
    with open("ctrl_P.dat", encoding='utf-8') as f:
        lines = f.readlines()
        P = [np.array(line.split(), dtype=np.float32) for line in lines[1:]]
    
    leading_curve = read_x("leading.x")[0].reshape([-1, 3])
    leading_P = P[0][:12].reshape([4, 3]).copy()

    P_fixed = leading_P
    x0 = np.hstack([P_fixed[1], P_fixed[2]])  # 初始值

    # 优化
    res = minimize(
        fun=loss,
        x0=x0,
        args=(P_fixed, leading_curve),
        method='L-BFGS-B',
        tol=1e-6
    )

    # 得到最优控制点
    P_opt = P_fixed.copy()
    P_opt[1] = res.x[:3]
    P_opt[2] = res.x[3:]
    print(P_opt, res.fun)
    coords_opt = Bezier_point(P_opt)

    # 把最优控制点塞回原 P
    P[0][:12] = P_opt.ravel()

    # 画图看效果
    plt.plot(leading_curve[:,0], leading_curve[:,1], 'b', label='target')
    plt.plot(coords_opt[:,0], coords_opt[:,1], 'r--', label='fitted')
    plt.legend()
    plt.show()
    # coords = Bezier_point(leading_P)
    # curve = draw_curve(P[0].reshape([-1, 3]))
    # redis(coords, 10)
    B = Bezier(P)
    B.write_curve([coords_opt], 'curve.x')
    # plt.plot(curve[:, 0], curve[:, 1])
    # plt.show()