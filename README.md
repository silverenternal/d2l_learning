# d2l - Rust 实现

使用 Rust 和 tch-rs 库实现《动手学深度学习》的部分练习。

## 环境配置

本地 libtorch 路径：`/opt/libtorch`

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
export LIBTORCH_BYPASS_VERSION_CHECK=1
```

## 程序说明

### ch2_5_2_autodiff - 自动微分基础

**功能**：演示 PyTorch 自动微分的基本用法

**思路**：
- 创建张量 `x = [0, 1, 2, 3]` 并启用梯度追踪
- 计算 `y = x²`
- 反向传播求梯度

**结果**：
```
梯度结果：[0.0, 2.0, 4.0, 6.0]  # dy/dx = 2x
```

---

### ch2_5_3_detach - 分离计算

**功能**：使用 `detach()` 分离计算图，阻止梯度回传

**思路**：
- 计算 `y = x²`，然后用 `u = y.detach()` 分离
- 计算 `z = u * x`，此时 u 被视为常数
- 反向传播验证：dz/dx = u（而非 3x²）
- 再对 y 反向传播验证：dy/dx = 2x

**结果**：
```
z 关于 x 的梯度 (u 视为常数): [0.0, 1.0, 4.0, 9.0]  # 等于 u
y 关于 x 的梯度 (y = x*x): [0.0, 2.0, 4.0, 6.0]     # 等于 2x
```

---

### ch2_5_5_sin_plot - sin 函数及其导数可视化

**功能**：绘制 f(x)=sin(x) 及其导数图像，不使用 cos 公式

**思路**：
- x 范围：[-2π, 2π]
- 自动微分：对 sin(x) 直接 backward() 求导
- 数值微分：中心差分法 f'(x) ≈ [f(x+h) - f(x-h)] / 2h，h=0.001
- 使用 plotters 绘制上下两图

**结果**：
- 输出 `sin_and_derivative.png`
- 两种微分方法结果重合，都得到 cos(x) 的波形

---

### house_price_preprocess - 房价预测数据预处理

**功能**：处理 Kaggle 房价预测数据集

**思路**：
1. 读取 train.csv（1460 样本 × 81 特征）
2. 统计缺失值，识别 19 个有缺失的特征
3. 识别离散型特征（56 个）
4. 填充缺失值：
   - 离散特征 → 众数填充
   - 连续特征 → 中位数填充
5. 删除 ID 列，保存处理后的数据

**结果**：
```
样本数：1460, 特征数：81

有缺失的特征 (19 个):
  - PoolQC: 1453 (99.5%)
  - MiscFeature: 1406 (96.3%)
  - Alley: 1369 (93.8%)
  - Fence: 1179 (80.8%)
  - FireplaceQu: 690 (47.3%)
  - ...

离散特征：56 个

已保存到：房价预测/train_processed.csv
处理后：80 特征，1460 样本
```

---

## 依赖

```toml
tch = "0.23"       # PyTorch Rust 绑定
plotters = "0.3"   # 绘图库
csv = "1.3"        # CSV 解析
serde = "1.0"      # 序列化
```

## 运行

```bash
# 自动微分
cargo run --bin ch2_5_2_autodiff

# 分离计算
cargo run --bin ch2_5_3_detach

# sin 函数绘图
cargo run --bin ch2_5_5_sin_plot

# 房价数据预处理
cargo run --bin house_price_preprocess
```
