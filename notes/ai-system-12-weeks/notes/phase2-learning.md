# 阶段 2 · 理解学习本身（第 4–6 周）

> 每章结构：**速览**（原推送要点）→ **深入与推导** → **代码实践** → **常见误区与面试考点** → **延伸阅读**。

---

## 第 4 周 · 损失函数

### 核心概念（速览）

**一句话**：损失函数是业务目标的数学翻译——你优化的不是你想要的，而是你写出来的；写错损失函数，模型会精确地优化错的东西。

$$L_{MSE} = \frac{1}{N}\sum(y_i - \hat{y}_i)^2 \qquad L_{MAE} = \frac{1}{N}\sum|y_i - \hat{y}_i| \qquad L_{CE} = -\sum_{c} y_c \log \hat{p}_c = -\log \hat{p}_{\text{true}}$$

**直觉类比**：损失函数是考试的评分标准。MSE 像"错得越多罚得越狠"，MAE 像"按错的比例扣分"，交叉熵像"对正确答案信心不足就扣分"——分类对了但概率只有 0.51 也照罚。

**关键要点**：①MSE 敏感离群点、MAE 鲁棒但梯度不平滑、Huber 折中；②CE 的梯度干净（∂L/∂z = p − y）源于 MLE；③损失 ≠ 评估指标，AUC/F1 不可导，训练用代理损失；④多任务总损失 ΣwᵢLᵢ，权重是业务优先级手柄。

### 推导：每个损失函数背后都有一个概率假设

**MSE ↔ 高斯噪声假设**（第 1 周已证）：p(y|x) = N(f(x), σ²)，MLE ⇒ 最小化 MSE。

**MAE ↔ 拉普拉斯噪声假设**：p(y|x) = Laplace(f(x), b) = (1/2b)·exp(−|y−f(x)|/b)，取负对数 ⇒ 最小化 MAE。

更深一层的结论：**MSE 回归的是条件均值 E[y|x]，MAE 回归的是条件中位数**。数据有重尾离群点时，均值被拖走、中位数不动——这就是"MAE 抗离群点"的概率本质。审核场景里，"审核耗时"类重尾目标用 MAE/Huber，"曝光-转化"类乘性噪声目标常先取 log 再回归。

**二元交叉熵 ↔ Bernoulli 假设**：p(y|x) = Bernoulli(σ(z))，负对数似然 = BCE。

### 推导：softmax + 交叉熵梯度的完整推导（必考）

设 logits z ∈ ℝᴷ，softmax：p_k = e^{z_k}/Σⱼe^{zⱼ}。损失 L = −log p_y（y 为真实类别）。

**第一步**：softmax 的导数（分 k=i 与 k≠i 两种情况，由商的求导法则）：

$$\frac{\partial p_k}{\partial z_i} = p_k(\delta_{ki} - p_i)$$

**第二步**：代入链式法则（注意 L 通过所有 p_k 依赖 z_i）：

$$\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial p_k}\cdot\frac{\partial p_k}{\partial z_i} = -\sum_k \frac{\delta_{ky}}{p_k}\cdot p_k(\delta_{ki} - p_i) = p_i - \delta_{iy}$$

即 **∂L/∂z = p − y（one-hot）**。三个重要推论：

1. **梯度只依赖预测分布与标签之差**，与 z 的绝对尺度无关——这是"干净"的数学含义；
2. **p = y 时梯度严格为 0**，损失同时取最小值——损失与梯度同零点，优化目标自洽；
3. **对错误分类的置信度惩罚是软性的**：p_true 从 0.4 提到 0.9 的梯度信号远强于从 0.01 提到 0.02——模型会优先"修好大致分得清的样本"。这正是 Focal Loss 想改造的性质（第 6 周）。

### 深入：交叉熵、KL 散度与信息熵的关系

$$H(p) = -\sum_c p_c \log p_c \qquad KL(p\|q) = \sum_c p_c \log\frac{p_c}{q_c} \qquad CE(p, q) = H(p) + KL(p\|q)$$

- **H(p)**：真实分布本身的"不确定性"（与模型无关的常数）；
- **KL(p‖q)**：用 q 近似 p 带来的额外编码长度——非负，仅当 p=q 时为 0，且**不对称**（KL(p‖q) ≠ KL(q‖p)）；
- 所以最小化 CE ≡ 最小化 KL ≡ 做 MLE，三者同解。

**面试常考**：为什么不对称？用 q 编码 p 的最短编码长度 ≠ 用 p 编码 q 的。"q 在 p 概率为 0 处给概率"会付出无穷代价（q_c=0 而 p_c>0 ⇒ KL=∞），所以 KL(p‖q)（前向 KL，覆盖模式）强迫 q 在 p 有质量处必须有质量——生成模型里 mode-covering vs mode-seeking 的区分源头。

### 深入：交叉熵的"信心惩罚"与温度

softmax 有温度参数：p_i = e^{z_i/T}/Σⱼe^{z_j/T}。

- T→0：分布退化为 one-hot（最自信/最锐）；
- T→∞：退化为均匀分布（最平）；
- T=1：标准 softmax。

交叉熵对"正确但低置信"的预测持续惩罚（−log 0.51 ≈ 0.67 仍然不小），所以训练充分的分类器往往**过度自信**——离线 AUC 不变但校准差。缓解手段见下文"温度缩放"。

### 深入：Huber 损失与分位数损失

Huber（δ 为拐点）：

$$L_\delta(e) = \begin{cases} \frac{1}{2}e^2 & |e|\le\delta \\ \delta(|e| - \frac{1}{2}\delta) & |e|>\delta\end{cases}$$

|e|<δ 时二次（平滑、随误差衰减梯度保证收敛精度），|e|>δ 时线性（梯度恒为 ±δ，不被离群点主导）。δ 的选取：δ = 1.345·σ（σ 为残差标准差估计）可在高斯数据上保持 95% 效率——工程上直接取目标分位点（如残差的 70% 分位）更省事。

**分位数损失（quantile / pinball loss）**：预测"第 τ 分位数"而非均值，审核排队时长预估等"要给安全上限"的场景天然适配：L_τ(e) = τ·max(e, 0) + (1−τ)·max(−e, 0)。

### 深入：多任务损失与权重设计

总损失 L = Σᵢ wᵢ·Lᵢ 中，wᵢ 怎么定？

- **手工调权**：按业务 KPI 优先级拍，再按各损失量级归一（先记录单任务训练的 loss 量级，避免某个任务凭量级主导梯度）；
- **不确定性加权（Kendall et al., 2018）**：把每个任务的观测噪声 σᵢ 也作为可学习参数，L = Σᵢ [Lᵢ/(2σᵢ²) + log σᵢ]——模型自动学会"哪个任务噪声大就少信它"，σᵢ 增大惩罚 log σᵢ 防止全丢；
- **动态加权（GradNorm/DWA）**：按各任务梯度量级平衡训练速度，避免某任务先收敛后主导。

工程经验：**先固定权重把流程跑通，再用验证集 A/B 细调权重——权重是杠杆最大的超参之一，但也是最容易过拟合验证集的东西**（每调一次都在泄漏一点验证信息）。

### 深入：校准（Calibration）——AUC 之外被忽视的指标

**校准的定义**：模型预测"概率 0.8"的样本集合里，真实正例比例应 ≈ 0.8。风控额度、审核置信度阈值决策都依赖概率值的含义，不只是排序。

度量：ECE（Expected Calibration Error，把预测概率分桶，比较每桶置信度与准确率的加权差）；可视化：reliability curve。

修正手段：

- **温度缩放（Temperature Scaling）**：在验证集上学习单一标量 T，推理时 z/T——最小化验证集 NLL 的一维优化，零训练成本，最常见的部署前校准；
- Platt scaling / isotonic regression：更多参数的校准（逻辑回归/单调回归拟合），小数据更稳。

### 工程场景（原文案例复盘）

推荐系统用 CTR 损失训练，上线后用户时长下降——点击率高的都是标题党。**根因**：代理指标偏离真实目标。**解法**：多目标建模（CTR + 时长 + 负反馈）或样本加权。

**复盘扩展**：这类"目标错配"事故的排查思路是固定的——①明确业务的真目标（留存/时长/GMV）；②列出训练目标与真目标的差异方向（点击↑ 与时长↓ 正相关于标题党内容）；③引入对齐信号（多目标、加权、或直接改标签定义：把"有效点击"做成标签）。**改标签定义往往比改模型结构便宜一个数量级**，例如把"点击"标签换成"点击且停留 > N 秒"。

### 常见误区与面试考点

1. **BCE 和 CE 的关系**：BCE 是 K=2 的交叉熵（标签 0/1，单输出 logit）；多分类 CE 用 K 个 logits + softmax。别把 BCE 用于多分类；
2. **"loss 在降但 AUC 不动"**：代理损失与排序指标出现分歧，检查是否存在大量"排序对"被损失忽视（如类别极不平衡时 loss 被易分样本主导 → Focal Loss）；
3. **面试常问**：softmax 导数？（p_k(δ_ki − p_i)）为什么分类用 CE 不用 MSE？（CE 梯度 p−y 不随误差变小而平方衰减、且与 sigmoid/softmax 的饱和区互补；MSE + sigmoid 时 sigmoid 饱和导致梯度消失，见第 2 周激活函数表）label smoothing 是什么？（y 从 one-hot 软化为 (1−ε)y + ε/K，防过度自信、提升校准与鲁棒，视觉/Transformer 训练常用 ε=0.1）

### 代码实践

```python
# 多分类 + 标签平滑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 温度缩放（推理前在验证集上拟合 T）
import torch
def fit_temperature(logits_val, y_val):
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    nll = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = nll(logits_val / T, y_val)
        loss.backward(); return loss
    opt.step(closure)
    return T.detach()
```

### 延伸阅读

- Goodfellow et al.《深度学习》第 3.13 节（信息论）、5.5 节（最大似然）
- Kendall, Gal & Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses* (CVPR 2018)
- Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017，温度缩放经典论文)

---

## 第 5 周 · 优化与泛化

### 核心概念（速览）

**一句话**：优化解决"训练误差怎么降下去"，泛化解决"测试误差怎么降下来"——两个独立的问题，会分别失败。

$$v_t = \beta v_{t-1} + \nabla L(\theta_t),\quad \theta \leftarrow \theta - \eta v_t$$

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\quad \theta \leftarrow \theta - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

**直觉类比**：SGD 蒙眼下山凭脚下坡度；动量加惯性——小坑洼不再改道、峡谷里不再震荡；Adam 给每个方向配独立步伐调节器。

**关键要点**：①SGD+momentum 泛化常最好但调参贵，Adam 上手稳是默认起点，AdamW 是 Transformer 标配；②病态曲率时朴素 SGD 沿垂直壁震荡；③偏差-方差分解指导先诊断再下药；④过参数化 + 隐式正则化解释了"能插值还泛化"的现代现象。

### 深入：凸优化速览——知道什么是不对的边界

机器学习优化的残酷前提：**深度网络的损失面非凸**。但凸优化的结论给直觉锚点：

- 凸函数：任一局部最小都是全局最小；SGD 在凸 Lipschitz 函数上平均迭代 O(1/√T) 收敛，光滑强凸 O(1/T)；
- 非凸世界这些保证全失效——存在鞍点、平坦/尖锐极小值、大量等价全局最优（参数置换对称性）；
- **好消息**：深度网络的坏情况（坏鞍点、平台区）实践中远没有理论最坏情况那么糟，且 SGD 的噪声本身帮助逃离鞍点（梯度近零处噪声成为主项，随机跳动后顺势下坡）。

### 推导：动量的两个等价视角

**视角一：EMA**。v_t = βv_{t−1} + (1−β)g_t（EMA 写法）是过去梯度的指数加权平均，β=0.9 时有效窗口 ≈ 1/(1−β) = 10 步。等价的"有效学习率"放大：匀速下降段 v ≈ g/(1−β)，步长 ≈ η/(1−β)——**β=0.9 让步长放大 10 倍**，这就是开动量后常常要降 lr 的原因。

**视角二： heavy ball ODE**。连续极限下 θ̈ + (a/η)θ̇ + ∇L = 0——带阻尼的质点在势能面滚落：惯性助其翻过窄脊与小坑，阻尼保证收敛。Nesterov 动量（NAG）是"先按惯性走到预估点再算梯度"，对这个物理图景做了预见性修正，凸情形下可证最优收敛率。

**什么时候动量最有用**：梯度噪声大（小 batch）、损失面呈长峡谷（目标相关性强的特征尺度悬殊）。动量几乎总是对的——不开动量才是需要理由的选择。

### 推导：Adam 完整机制与偏差校正

三个步骤：

1. **一阶矩** m_t = β₁m_{t−1} + (1−β₁)g_t（梯度的 EMA，相当于动量）；
2. **二阶矩** v_t = β₂v_{t−1} + (1−β₂)g_t²（梯度**平方**的 EMA，估计各参数的梯度尺度）；
3. **更新** θ ← θ − η·m̂_t/(√v̂_t + ε)。

**偏差校正**：m_0 = v_0 = 0 导致初期估计系统性偏小（偏向 0），除以 (1−βᵗ) 校正：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t},\qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

t=1 时 m̂₁ = g₁、v̂₁ = g₁²，完全无偏——之后随 t 增大校正因子趋近 1。β₂=0.999 是常见默认，t 较小时 (1−β₂ᵗ) 很小，校正在前几百步作用显著。

**每参数自适应步长的意义**：把更新归一化到大致 ±η 的量纲，**不同尺度参数的更新速度自动均衡**——这就是为什么 Adam 对学习率远没 SGD 敏感、是默认起点。

### 推导：AdamW——为什么权重衰减要"解耦"

经典 L2 正则：在损失里加 (λ/2)‖θ‖²，梯度变成 g + λθ。**问题**：Adam 的更新 ∝ m̂/√v̂，把 L2 项混入梯度后，正则力度会被二阶矩归一化扭曲——对梯度常年小的参数，λθ 项在 √v̂ 分母下被不成比例地缩小，正则失效；AdamW 把衰减从梯度中拿出，直接放在更新步上：

$$\theta \leftarrow \theta - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta$$

**实践结论：Transformer 及几乎所有现代场景用 AdamW 而非 Adam + L2**，`torch.optim.AdamW(weight_decay=0.01~0.1)`。

### 深入：学习率调度的原理（第 6 周详表）

大方向学习率必须衰减的直觉：优化前期要探索（大步长逃离坏区域、平均掉噪声），后期要精修（小步长压优化间隙）。两种经典理论视角：①SGD 收敛界里误差项 ∝ η，后期缩小 η 直接压掉随机误差项；②把后期训练看作"在好极小值附近平均化噪声"，小 lr 提高平均精度。具体 schedule（step/cosine/warmup 公式与选择）见第 6 周①。

### 深入：二阶方法——知道名字就够，但要知道为什么不用

- **牛顿法**：θ ← θ − H⁻¹∇L，H 为 Hessian。收敛快（二次收敛）但 H 的存储 O(n²)、求逆 O(n³)，n=10⁹ 直接出局；
- **拟牛顿 L-BFGS**：不显式存 H，存少量向量对近似 H⁻¹——中小模型（如校准里的 T 优化、逻辑回归）仍是首选；`torch.optim.LBFGS` 但注意它需要 full-batch 或闭合函数；
- **Adam ≈ 对角 Hessian 近似的随机版**：√v̂ 粗略扮演曲率的角色（梯度平方大 ≈ 曲率大）。这是 Adam 族与二阶方法的概念桥梁，也是理解其局限（对角近似忽略参数间曲率耦合）的钥匙。

### 推导：偏差-方差分解

对回归（平方损失），期望预测误差可分解：

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f(x)\big)^2}_{\text{偏差}^2} + \underbrace{\mathbb{E}\big[\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\big]^2}_{\text{方差}} + \underbrace{\sigma^2}_{\text{不可约噪声}}$$

- **高偏差**：训练集都拟合不好（欠拟合）——加容量、训更久、减正则；
- **高方差**：训练好验证差（过拟合）——加数据、加正则、降容量、早停；
- 学习曲线（train/val error 随数据量/epoch 变化）是诊断偏差的金标准。

### 深入：现代泛化之谜

经典理论预言：参数 ≫ 数据 ⇒ 过拟合 ⇒ U 形风险曲线。实践观察（Belkin et al. 2019 系统化）：插值阈值之后，风险**再次下降**（double descent）——过参数化 + 适当正则的模型反而最优。

主流解释线索：

- **隐式正则化/隐式偏差**：SGD（尤其小 batch）偏好平坦极小值；过参数化空间里"拟合训练集的方式"有无穷多种，优化算法暗中选择了泛化好的那批；
- **平坦极小值与泛化相关**（直觉：平坦 ⇒ 对参数扰动/数据扰动不敏感 ⇒ 测试误差近似训练误差）；SAM（锐度感知最小化）把"找平坦解"显式化；
- **神经正切核（NTK）**视角：过参数化极限下网络等价于核方法，可证泛化界。

工程含义：**"过参数化不是罪，正则与优化动态才决定泛化"**——这与第 6 周的实践清单一致。

### 工程场景（原文案例复盘）

风控模型季度迭代：训练 AUC 0.92、上线 0.78。诊断先查分布偏移——训练样本是 3 个月前的欺诈模式，黑产已换代。

**复盘扩展**：离线-线上指标差的完整排查顺序（按成本从低到高）：

1. **评估口径**：线上指标的计算方式与离线是否一致（时间窗、标签延迟——风控里"未逾期"≠"不逾期"，需表现期）；
2. **分布偏移**：特征 PSI、标签先验漂移、时间外推性验证（用最后 N 天做 holdout，而非随机切分）；
3. **数据泄漏**：特征里是否有线上不可得的字段（如用"处置结果"反推）；
4. **训练问题**：学习率/正则不当（双下降时代的"训练误差 100%"也不等于学好了——检查 val 曲线而非 train）。

PSI 公式（Population Stability Index，漂移监控标准工具）：

$$PSI = \sum_i (a_i - e_i)\,\ln\frac{a_i}{e_i}$$

a_i = 当前分布第 i 桶占比，e_i = 基准分布第 i 桶占比。< 0.1 稳定，0.1–0.25 需关注，> 0.25 显著漂移。

### 常见误区与面试考点

1. **"Adam 一定比 SGD 好"**——收敛快 ≠ 泛化好；视觉精调任务上 SGD+momentum 常有 1 个点左右优势，Adam 赢在稳健与速度；
2. **"loss 降到 0 说明过拟合了"**——插值训练集在现代 regime 下未必坏（见上），要看验证曲线；
3. **面试常问**：Adam 的 ε 是干嘛的？（分母数值保护，防梯度为 0 时除零）为什么 β₂ 常用 0.999？（二阶矩变化慢需长窗口）SGD 噪声从哪来？（mini-batch 采样，方差 ∝ (1/B − 1/N)，第 6 周③）鞍点和局部极小值哪个更麻烦？（高维里鞍点更常见——Hessian 所有维度同号的概率随维数指数衰减）

### 延伸阅读

- Kingma & Ba, *Adam: A Method for Stochastic Optimization* (ICLR 2015，原文)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (ICLR 2019，AdamW)
- Belkin et al., *Reconciling modern machine-learning practice and the classical bias–variance trade-off* (PNAS 2019，双下降)
- Foret et al., *Sharpness-Aware Minimization* (ICLR 2021，SAM)

---

## 第 6 周 · 训练工程五件套

### ① 学习率调度

**速览**：学习率是最重要的超参——太大震荡发散，太小收敛慢/卡鞍点。warmup（开局小步预热）→ 衰减（余弦/阶梯/线性）。

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})\left(1+\cos\frac{t\pi}{T}\right)$$

**深入——每种 schedule 的性格**：

| 调度 | 特点 | 适用 |
|---|---|---|
| 固定 | 基准线 | 几乎不用（除非配合好的自适应优化器） |
| 阶梯衰减（step） | 每 N epoch ×0.1 | 传统 CV 训练（ResNet 官方配方），最终 lr 极小利于精修 |
| 余弦退火 | 平滑降到 0 | Transformer/大模型标配；尾部小 lr 时间占比高 ≈ 自带"末期精修" |
| 线性衰减 | 简单、末段留非零底 | 大模型预训练常用（配 warmup） |
| 循环（SGDR/cosine restart） | 周期性重启大 lr | 需要多次逃离局部结构时（超参搜索/集成） |

**warmup 为什么几乎必需**（尤其 Adam + Transformer）：初始化时权重远离任何好解，梯度信号强且 v̂ 估计不准（偏差校正初期），大 lr 会直接爆炸；前几百~几千步线性/线性升到 η_max 让二阶矩估计稳定下来。经验规则：warmup 步数 ≈ 总步数的 1–10% 或 5,000 步内（大模型取大值）。

**初值选取**：①有现成 recipe 就抄（同架构同量级）；②网格粗搜 {1e-5, 3e-5, 1e-4, 3e-4}（Adam 系）；③训练曲线判断：loss 前几百步就 NaN → 降 10 倍；几百 epoch 纹丝不动 → 升 10 倍。

### ② 类别不平衡

**速览**：欺诈/违规 <1% 时朴素训练被多数类淹没。武器库：重采样（SMOTE）、代价敏感（w_c ∝ 1/freq_c）、Focal Loss、PR 曲线/AUPRC 评估；阈值按业务成本矩阵定，不用默认 0.5。

**深入——四种武器的原理与选择**：

**a) 重采样**：

- 过采样（重复少数类）：零信息损失，但重复样本 → 易对少数类过拟合（需配合更强正则）；
- SMOTE：对少数类样本 x 与其近邻 x̂ 线性插值生成 x_new = x + λ(x̂−x)，λ∈(0,1)。缺陷：高维/多模态数据可能生成"缝合怪"样本；对文本需嵌入空间插值；
- 欠采样（砍掉多数类）：快，但丢信息；常用于"多数类远多于所需"（千万级负样本留十万）；
- **动态采样（dynamic class sampling）**：每个 epoch 按当前各类别难度重新决定采样率——比静态比例更稳。

**b) 代价敏感（类别权重）**：损失乘 w_c ∝ 1/freq_c。等价于对少数类**重要性采样**——数学上与过采样的期望梯度相同，实现上零数据搬运，**通常首选**。注意：权重过大（如 w=1000）会导致少数类样本主导梯度、loss 震荡——配合梯度裁剪。

**c) Focal Loss**（RetinaNet, ICCV 2017）：

$$FL(p_t) = -\alpha(1-p_t)^\gamma \log p_t$$

(1−p_t)^γ 是**调制因子**：易分样本（p_t→1）因子→0 自动降权，难分样本（p_t 小）因子≈1 保留全部梯度。效果：交叉熵损失被海量"随手答对的多数类样本"稀释的问题被显式修复，梯度聚焦在难例上。γ=2、α=0.25 是常用起点。缺陷：对噪声标签敏感（难例里混着标错的样本会被加倍关注）。

**d) 阈值与评估**：训练时 imbalance 用 a/b/c 解决，**决策时用阈值把分类器"翻译"回业务**：按成本矩阵 C_FN/C_FP 选阈值使期望成本最小（公式见第 1 周），并在验证集上按线上约束（如"人工复审队列容量 ≤ 5%"）反解阈值。评估：AUPRC、Recall@固定 FPR、PR 曲线——不平衡下准确率与 ROC-AUC 都会说谎。

### ③ batch size 与梯度噪声

**速览**：batch size 决定梯度信噪比。小 batch 噪声大有正则效应；大 batch 稳但需更大 lr 与 warmup，否则泛化下降。梯度噪声尺度 ≈ η(1/B − 1/N)。

**深入——线性缩放规则的推导直觉**：SGD 更新噪声（单步）方差 ∝ η²/B。要让大批量 B'=kB 保持同样的"噪声强度"，η' = kη。即 **batch 翻 k 倍、lr 翻 k 倍**（线性缩放规则，Goyal et al. 2017），让训练初期动力学近似不变。注意该规则适用于 SGD/动量的早期训练阶段 + 配合 warmup；超大 batch（>8k）需要 LARS/LAMB（逐层自适应 lr）等专门优化器。

**深入——batch size 与泛化的权衡表**：

| | 小 batch（≤64） | 大 batch（≥1k） |
|---|---|---|
| 梯度 | 噪声大 | 稳 |
| 泛化 | 常更好（噪声隐式正则） | 需调 lr/warmup 才不掉 |
| 吞吐 | 硬件利用率低 | 高（GPU 吃满） |
| 调参 | 对 lr 相对鲁棒 | 敏感，缩放规则必配 |

**临界 batch size**（McCandlish et al. 2018）：继续加大 batch 不再减少总训练时间的拐点，受梯度噪声尺度控制。工程含义：不要无限制堆大 batch——超过临界值只省时间不涨质量，还占显存。

**显存不够时的三板斧**：①梯度累积（micro-batch 前向反向 N 次再 step 一次，等价大 batch）；②混合精度训练（AMP：fp16/bf16 前向反向，fp32 存主权重，显存近半 + 速度快）；③梯度检查点（重算前向换显存，见第 3 周）。

### ④ 正则化与早停

**速览**：正则化 = 给假设空间加约束，牺牲训练拟合换泛化。L2/L1/Dropout/数据增强/早停。

**深入——L2 的约束优化视角**：min L(w) s.t. ‖w‖² ≤ c，其拉格朗日形式 min L(w) + λ‖w‖²——**带惩罚的 L2 与带约束的 L2 同解**（λ 与 c 单调对应）。直觉：把解限制在小球内 = 限制有效假设空间。L1 的约束是多面体，顶点在坐标轴上 ⇒ 稀疏解（部分权重严格为 0）→ 可做特征选择；L2 的约束是球 ⇒ 权重小而分散。

**深入——L2 vs 权重衰减（再次强调）**：SGD 下两者等价；Adam 下**不等价**（AdamW 才正确实现"想让正则起的作用"）——第 5 周推导过。

**深入——Dropout 的正确理解**：训练时以概率 p 随机置零，**存活神经元输出 ×1/(1−p)**（inverted dropout），使输出期望不变；推理时不做任何事（开关全闭合的"完整网络"）。

- 近似视角：每次 step 采样一个子网络，训练 ≈ 指数级子网络的**模型集成**的分布式近似；
- 共适应视角：防止神经元"抱团"（一个神经元修正另一个的错），迫使每个特征单独有用——这解释了为什么 Dropout 在特征高度冗余的层（大 FC）最有效，在 BN 层附近要小心（方差漂移，通常 Dropout 放在 BN 之前或不共用）；
- 现代趋势：视觉里被数据增强部分替代，Transformer 里常用 DropPath（随机丢整层）。

**深入——数据增强为什么是最便宜的正则**：它等价于告知模型"标签在这类变换下不变"（翻转/裁剪不变、同义替换不变）= 注入正确的归纳偏置，同时等效扩充数据。增强策略应与部署分布一致（审核场景慎用过度裁剪：主体占比本身可能是判别特征——**增强策略也是先验，选错就是教错**）。

**深入——早停作为正则的机理**：验证损失上升再停，实质是把"训练步数"变成了容量约束——停在泛化最优的路径中段，等价于限制了解所在的空间区域。工程要点：patience 与监控指标的选择（监控业务指标而非 val loss 更直接）、恢复 best checkpoint、cosine 这类自带末期小 lr 的 schedule 与早停配合时注意"还没到收敛就停"的误停。

**其他现代正则（知道名字）**：Mixup（x、y 线性混合 ⇒ 样本间插值平滑）；CutMix（区域拼接 + 标签按比例混合）；label smoothing（第 4 周）；R-Drop/一致性正则（同输入两次前向输出一致）。

### ⑤ 初始化与归一化

**速览**：深度网络能训练的前提是前向信号与反向梯度各层不衰减不爆炸。Xavier/He 初始化对齐方差；BatchNorm 平滑损失面；LayerNorm 不依赖 batch 是 Transformer 标配。

$$BN(x) = \gamma\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}} + \beta$$

**推导：Xavier/He 的方差传播推导**（必考）：设输入 x 各维独立零均值方差 Var(x)，权重独立零均值，则 y = Σᵢ wᵢxᵢ 的方差：

$$Var(y) = \sum_{i=1}^{n_{in}} Var(w_i)\,Var(x_i) = n_{in}\,Var(w)\,Var(x)$$

前向信号不衰减 ⇒ Var(y)=Var(x) ⇒ **Var(w) = 1/n_in**。反向同理要求 Var(w) = 1/n_out。Xavier 取折中：Var(w) = 2/(n_in+n_out)。

He 初始化：ReLU 把一半激活置 0，有效方差减半，补偿因子 2：**Var(w) = 2/n_in**（ReLU 配 He，tanh/sigmoid 配 Xavier——这是面试速配题）。

**深入——BatchNorm 的三件事**：①归一化（减均值除方差）；②再仿射（可学习 γ、β——恢复表达能力，极端情况可学回恒等）；③推理时用训练期滑动平均的 μ、σ（**train/eval 统计量不同**是 BN 的典型坑，见第 3 周）。

为什么 BN 有效（多重解释，都对一部分）：①平滑损失面（lipschitz 常数改善）→ 允许大 lr；②减弱协变量偏移（后续层输入分布稳）；③噪声正则（batch 统计量本身是噪声）；④让损失面曲率更均匀（与 Adam 的自适应步长互补）。

BN 的坑：①小 batch（<8）统计量噪声大 → 换 GroupNorm/LayerNorm；②batch 间样本分布不均（如长序列 padding）→ 统计量被污染；③train/eval 不一致导致的推理抖动（严格时应固定统计量重校）。

**深入——归一化方法速查表**：

| 方法 | 归一化维度 | 依赖 batch | 典型场景 |
|---|---|---|---|
| BatchNorm | 单通道，跨 batch 和空间 | ✅（缺点） | CNN 视觉 backbone |
| LayerNorm | 单样本，跨特征 | ❌ | Transformer、RNN |
| RMSNorm | LN 去掉均值中心化（只除 RMS） | ❌ | LLaMA 系（省一次统计量） |
| GroupNorm | 通道分组内 | ❌ | 小 batch 检测/分割 |
| InstanceNorm | 单样本单通道 | ❌ | 风格迁移 |

Transformer 内部还有一个结构选择：**Post-Norm vs Pre-Norm**。原始 Transformer 是 Post-Norm（x + Sublayer(x) 再 LN），深层时梯度路径含 LN 的非线性、训练易不稳；现代大模型多用 Pre-Norm（x + Sublayer(LN(x))），每层梯度高速路更干净、可训更深，但需注意 final LN 的放置与小幅精度差异。面试高频。

### ⑥ 第 6 周串联复习：统一视角 + 实战清单

**统一视角**（原文摘要）：五件套都在控制"有效复杂度"与"优化稳定性"——学习率调度/batch size 管优化动态；类别不平衡管数据分布与目标的匹配；正则化/归一化管假设空间与信号传播。

**拿到任务的实战决策流**（把六周串成一张 checklist）：

1. **看数据**：类别比例？时间结构（按时间切分）？离群点？标签噪声？→ 决定采样、权重、损失选型；
2. **定评估**：业务指标（Recall@FPR、AUPRC…）与切分方案先写死在配置里，防止事后"调指标"；
3. **选基线**：逻辑回归/XGBoost 先跑通，深度学习方案必须证明相对基线的增量；
4. **配训练**：AdamW + warmup(数百~数千步) + cosine；梯度裁剪 1.0；监控 grad_norm 与 train/val 曲线；
5. **做诊断**：指标缺口 → 偏差（容量/训练不足）/方差（数据/正则）/偏移（时间外推验证、PSI 监控）三分法对症下药；
6. **上工程**：阈值按成本矩阵定、部署前温度缩放、上线后 PSI + 指标监控、灰度 + 回滚兜底。

### 常见误区与面试考点（六周合集）

1. **L1 与 L2 的几何差异**（菱形约束 vs 圆形约束 → 稀疏 vs 小权重）、L1 不可导点如何处理（次梯度）；
2. **BN 在推理时用什么的统计量**（running mean/var，训练期滑动平均）；
3. **为什么大 batch 要 warmup + 大 lr**（线性缩放规则 + 初期动力学匹配）；
4. **Focal Loss 的 γ 调大会怎样**（更聚焦极难样本，但更易被噪声标签带偏）；
5. **weight decay 与 L2 何时不等价**（自适应优化器 Adam 下）；
6. **梯度裁剪的位置**（backward 之后、step 之前，按全局范数 clip）与作用（防爆、不改变方向只缩尺度）。

### 代码实践

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
steps_per_epoch = len(train_loader)
total_steps = epochs * steps_per_epoch
warmup = min(500, total_steps // 10)

def lr_lambda(step):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog))          # 余弦退火到 0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

for x, y in train_loader:
    with torch.autocast('cuda', dtype=torch.bfloat16):     # 混合精度
        loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()                                       # 每个 step 更新
```

### 延伸阅读

- Goyal et al., *Accurate, Large Minibatch SGD* (arXiv 2017，线性缩放规则)
- Lin et al., *Focal Loss for Dense Object Detection* (ICCV 2017)
- Ioffe & Szegedy, *Batch Normalization* (ICML 2015)；Ba et al., *Layer Normalization* (2016)；Wu & He, *Group Normalization* (ECCV 2018)
- Srivastava et al., *Dropout* (JMLR 2014)；Zhang et al., *Mixup* (ICLR 2018)
- McCandlish et al., *An Empirical Model of Large-Batch Training* (2018，临界 batch size)
