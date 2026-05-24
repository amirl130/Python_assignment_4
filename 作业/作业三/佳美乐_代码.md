# ============================================
# 作业3：金砖国家FDI与经济增长关系分析
# 成员C：佳美乐 — 热力图 + 3D回归曲面图
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('brics_fdi_timeseries.csv')
print("数据加载成功！")
print(df.head())

# ============================================
# 可视化①：热力图（相关系数矩阵）
# ============================================

print("\n" + "="*50)
print("正在生成热力图...")
print("="*50)

# 选择数值列
数值列 = ['GDP_growth', 'FDI_inflow', 'GDP_percapita']
相关系数矩阵 = df[数值列].corr()

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(相关系数矩阵, 
            annot=True,           # 显示数值
            fmt='.2f',            # 保留2位小数
            cmap='RdBu_r',        # 红蓝配色
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('图②：相关系数热力图\n金砖国家（2010-2019年）', fontsize=14)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 热力图已保存为 heatmap.png")

# ============================================
# 可视化②：3D回归曲面图
# ============================================

print("\n" + "="*50)
print("正在生成3D回归曲面图...")
print("="*50)

# 准备数据
X1_FDI = df['FDI_inflow'].values           # FDI流入量
X2_人均GDP = df['GDP_percapita'].values     # 人均GDP
Y_增长 = df['GDP_growth'].values            # GDP增长率

# 添加常数项（截距）
X = np.column_stack((np.ones(len(X1_FDI)), X1_FDI, X2_人均GDP))

# 使用最小二乘法计算回归系数
回归系数, 残差, 秩, 奇异值 = np.linalg.lstsq(X, Y_增长, rcond=None)
截距, FDI系数, 人均GDP系数 = 回归系数

print(f"\n回归方程: GDP增长率 = {截距:.3f} + {FDI系数:.3f}×FDI + {人均GDP系数:.6f}×人均GDP")

# 创建网格用于绘制曲面
x1网格 = np.linspace(X1_FDI.min(), X1_FDI.max(), 30)
x2网格 = np.linspace(X2_人均GDP.min(), X2_人均GDP.max(), 30)
X1网格, X2网格 = np.meshgrid(x1网格, x2网格)
Y曲面值 = 截距 + FDI系数 * X1网格 + 人均GDP系数 * X2网格

# 国家颜色映射
国家颜色 = {
    'China': 'red',       # 中国 - 红色
    'India': 'orange',    # 印度 - 橙色
    'Brazil': 'green',    # 巴西 - 绿色
    'South Africa': 'blue' # 南非 - 蓝色
}

# 创建3D图形
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点（真实数据）
for 国家 in df['Country'].unique():
    子集 = df[df['Country'] == 国家]
    ax.scatter(子集['FDI_inflow'], 
               子集['GDP_percapita'], 
               子集['GDP_growth'],
               c=国家颜色.get(国家, 'gray'),
               s=80,
               label=国家,
               edgecolors='black',
               linewidth=1,
               alpha=0.8)

# 绘制回归曲面
ax.plot_surface(X1网格, X2网格, Y曲面值, 
                alpha=0.4, 
                color='gray',
                edgecolor='none')

# 设置标签
ax.set_xlabel('FDI流入量（占GDP百分比）', fontsize=12)
ax.set_ylabel('人均GDP（PPP，国际美元）', fontsize=12)
ax.set_zlabel('GDP增长率（%）', fontsize=12)
ax.set_title('图③：3D回归曲面图\nFDI与人均GDP对GDP增长的联合影响', fontsize=14)

# 设置最佳视角
ax.view_init(elev=25, azim=45)

# 显示图例
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('3d_regression.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 3D曲面图已保存为 3d_regression.png")

# ============================================
# 输出相关系数矩阵
# ============================================
print("\n" + "="*50)
print("相关系数矩阵：")
print("="*50)
print(相关系数矩阵.round(3))

print("\n" + "="*50)
print("✅ 全部完成！")
print("生成文件：heatmap.png, 3d_regression.png")
print("="*50)
