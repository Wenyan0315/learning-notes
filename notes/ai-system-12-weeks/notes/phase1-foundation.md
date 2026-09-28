# 阶段 1 · 建立直觉（第 1–3 周）

> 每章结构：**速览**（原推送要点）→ **深入与推导** → **代码实践** → **常见误区与自测要点** → **延伸阅读**。

---

## 第 1 周 · 从规则到学习：机器学习是什么

### 核心概念（速览）

**一句话**：机器学习不是写规则，而是让程序从数据中**学出规则**——用参数化函数 f(x; θ) 逼近真实的输入→输出映射，θ 由数据而不是工程师决定。

**三要素形式化**：

- **假设空间** H：候选函数族（如所有线性函数 wx+b）；
- **损失函数** L：衡量预测与真实的差距；
- **优化算法**：在 H 中找使损失最小的 θ。

经验风险最小化（ERM）：

$$\theta^* = \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} L(f(x_i;\theta), y_i)$$

**直觉类比**：传统规则引擎像老师傅把所有经验写成操作手册——手册永远写不完；机器学习像让徒弟自己看一万个案例总结规律——能泛化到没见过的案例，但你没法逐条审阅他的"脑内规则"，只能用考试（测试集）检验。

**关键要点**：①规则脑输出 if-then，概率脑输出 P(y|x)，阈值是两脑之间的翻译器；②训练/验证/测试三分，测试集泄漏是新手最常犯的评估事故；③泛化是唯一的考试，过拟合 = 背题；④没有免费午餐定理：选模型 = 选归纳偏置。

### 深入：学习的类型与各自的位置

| 类型 | 数据形式 | 审核/风控场景示例 | 代表方法 |
|---|---|---|---|
| 监督学习 | (x, y) 成对 | 违规分类、欺诈二分类、CTR 预估 | 线性模型、树模型、深度网络 |
| 无监督学习 | 只有 x | 黑产团伙聚类、异常交易检测 | K-means、孤立森林、自编码器 |
| 自监督学习 | 从 x 构造伪标签 | 预训练审核文本编码器（MLM） | BERT、对比学习 |
| 强化学习 | 状态-动作-奖励序列 | 审核队列的处置策略调度 | Q-learning、PPO |

工程上成熟的混合系统几乎都是"监督模型为主干 + 规则兜底"，原因不是 RL/无监督不先进，而是**监督学习的目标函数与业务 KPI 最容易对齐**（这是第 4 周的伏笔）。

### 推导：ERM 与极大似然估计（MLE）是同一枚硬币

最小二乘线性回归 = "噪声是高斯分布"假设下的 MLE：

$$p(y|x;\theta) = \mathcal{N}(f(x;\theta),\ \sigma^2) \;\Rightarrow\; \log p(\text{data}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - f(x_i;\theta))^2$$

最大化对数似然 ≡ 最小化 MSE（σ² 是常数）。同理，逻辑回归 = Bernoulli 噪声下的 MLE：

$$p(y|x) = \text{Bernoulli}(\sigma(w^\top x)) \;\Rightarrow\; \log p(\text{data}) = \sum_i \big[\, y_i \log p_i + (1-y_i)\log(1-p_i) \,\big]$$

最大化它 ≡ 最小化二元交叉熵。**结论：每个损失函数背后都有一个"噪声模型"假设；写损失函数 = 假设数据是怎么噪声化的。** 这个视角在第 4 周会反复用到。

### 深入：评估指标详解（工程高频考点）

混淆矩阵（以欺诈检测为例）：

| | 预测欺诈 | 预测正常 |
|---|---|---|
| **实际欺诈** | TP | FN（漏放，代价最高）|
| **实际正常** | FP（误伤用户）| TN |

- **Precision** = TP/(TP+FP)：被判为欺诈的人里有多少是真的——决定误伤率；
- **Recall（查全率）** = TP/(TP+FN)：真欺诈里抓到了多少——决定漏放率；
- **F1** = 2PR/(P+R)：两者的调和平均；
- **ROC-AUC** = P(score(随机一个正样本) > score(随机一个负样本))——概率解释，对类别比例不敏感；
- **PR-AUC / AUPRC**：不平衡场景的可靠指标（基线是正样本比例，不是 0.5）；
- **KS** = max |F_正(score) − F_负(score)|——风控评分卡传统指标。

**AUC 的概率解释是最常考的问法**，记住：AUC = 随机抽一对（正，负）样本，正样本得分更高的概率。

### 深入：数据划分的工程细节

- **K 折交叉验证**：小数据集上调超参的标准做法；模型最终要在全部训练数据上重训；
- **分层采样（stratified）**：保证每折的类别比例与全量一致，不平衡数据必用；
- **时间序列切分**：风控/推荐必须按时间切（用过去预测未来），随机切分会泄漏未来信息， offline AUC 虚高；
- **数据泄漏的隐蔽形式**：特征里混入"事后才知道"的信息（如用"是否投诉"预测"是否欺诈"——投诉发生在处置之后）；同一用户/设备出现在训练集和测试集（按 entity 去重分组切分）；预处理在全量数据上 fit（正确的做法是在每折的训练折内 fit scaler）。

### 工程场景（扩展）

审核系统典型混合架构：规则层（黑名单、敏感词，确定性兜底）→ 模型层（概率打分）→ 阈值决策（通过 / 人工复审 / 拒绝）。

阈值决策可以形式化为成本矩阵问题：设漏放成本 C_FN、误伤成本 C_FP，则最优阈值 t\* 满足

$$t^* = \frac{C_{FP}}{C_{FP} + C_{FN}} \quad\text{（先验均衡、模型校准良好的近似）}$$

引入"人工复审"本质是**拒绝选项（abstention）**：模型对 p∈[t_low, t_high] 的不确定样本不自动决策，转人工。这在数学上等价于一个三路分类器，用一点人工成本换漏放率大幅下降——混合系统的"人机协同"不是妥协，是有理论依据的最优结构设计。

### 常见误区与自测要点

1. **"训练集表现好 = 模型好"**——错，只看测试集/线上表现；
2. **准确率为什么不靠谱**：欺诈占比 0.1% 时，全猜"正常"就有 99.9% 准确率；
3. **AUC 高 ≠ 校准好**：AUC 只关心排序，不关心概率值是否等于真实频率（校准要用 reliability curve / ECE，第 4 周展开）；
4. **常见追问**：偏差-方差是什么？（第 5 周详解）L1/L2 区别？（第 6 周）怎么处理类别不平衡？（第 6 周）

### 延伸阅读

- 《统计学习方法》李航，第 1 章（统计学习三要素的形式化）
- Domingos, "A Few Useful Things to Know about Machine Learning"（经典短文）
- Google ML Rules（Martin Zinkevich, 2016，工程规则集）

---

## 第 2 周 · 神经网络基础：从感知机到 MLP

### 核心概念（速览）

**一句话**：神经网络 = 线性变换 + 非线性激活的反复堆叠，深度让网络用**组合式特征层次**表达复杂函数，参数量却远小于穷举。

$$z = w^Tx + b,\quad a = \sigma(z) \qquad\qquad h^{(l)} = \sigma(W^{(l)}h^{(l-1)} + b^{(l)}),\ l=1,\dots,L$$

**直觉类比**：单层感知机只会"画一条直线切蛋糕"；加非线性并堆叠后变成"先切小块再拼出任意形状"的老师傅。没有激活函数，100 层线性堆叠等价于 1 层——非线性是深度的意义所在。

**关键要点**：①激活函数谱系 sigmoid/tanh → ReLU → LeakyReLU/GELU；②通用近似定理："可以逼近" ≠ "好优化"，深度的价值是参数效率；③特征层次是 CNN 和 NLP 共通的组织原则；④输出层必须与损失配套（softmax + 交叉熵是标准组合）。

### 深入：从感知机到 XOR——为什么必须非线性

感知机（1958）是线性分类器：sign(wᵀx + b)。1969 年 Minsky 指出**感知机无法表示 XOR**——XOR 的四个样本点 (0,0)→0, (1,1)→0, (0,1)→1, (1,0)→1 是线性不可分的。

两层网络可以解决：

$$h_1 = \text{ReLU}(x_1 + x_2 - 0.5),\quad h_2 = \text{ReLU}(x_1 + x_2 - 1.5),\quad y = \text{ReLU}(h_1 - h_2 - 0.5)$$

h₁ 实现 OR，h₂ 实现 AND，y = OR − AND = XOR。**历史意义**：这个"缺陷"让神经网络研究停滞近 20 年，直到反向传播（1986）证明多层网络可训练。教训：一个模型的"表达能力下限"和"可训练性上限"是两件事——这个主题会一路伴随我们到 Transformer。

### 深入：激活函数详解

| 激活 | 公式 | 导数 | 优点 | 缺点 |
|---|---|---|---|---|
| sigmoid | σ(z)=1/(1+e⁻ᶻ) | σ(1−σ) | 概率解释 | 饱和区梯度≈0；输出非零中心 |
| tanh | (eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ) | 1−tanh² | 零中心 | 仍饱和 |
| ReLU | max(0, x) | 𝟙[x>0] | 简单、不饱和、计算快 | 神经元死亡（x<0 梯度恒 0）；输出非零中心 |
| LeakyReLU | max(αx, x), α≈0.01 | α 或 1 | 修复死亡 | α 需调 |
| GELU | x·Φ(x)（Φ 为标准正态 CDF） | Φ(x)+xφ(x) | 平滑、Transformer 标配 | 计算略贵 |

**GELU 直觉**：不是"硬切零"，而是按输入大小**随机软门控**——x 越小越可能被乘以接近 0 的系数。BERT/GPT 都用它。

**死亡 ReLU 的诊断**：训练中出现大量激活恒为 0 的神经元（histogram 里一坨死在 0），学习率过大 + 负偏置初始化是常见诱因；LeakyReLU/ELU 是缓解手段，但实践中"把学习率调小 + 用 He 初始化"往往就够了。

### 推导：MLP 的反向传播完整公式（第 3 周的预习）

记号：第 l 层 z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾+b⁽ˡ⁾，a⁽ˡ⁾ = σ(z⁽ˡ⁾)，损失 L。定义误差项 δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾。

输出层（以 softmax+CE 为例，第 4 周会推导）：δ⁽ᴸ⁾ = p − y。

隐层递推：

$$\delta^{(l)} = \big(W^{(l+1)}\big)^\top \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

参数梯度：

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \big(a^{(l-1)}\big)^\top, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

矩阵形状自检：δ⁽ˡ⁾ 是 (n_l, 1)，a⁽ˡ⁻¹⁾ 是 (n_{l-1}, 1)，外积得 (n_l, n_{l-1}) = W⁽ˡ⁾ 的形状。**形状对不上 = 推导错了**，这是手推时最快的自查方法。

### 代码实践：最小 MLP（PyTorch）

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in=20, d_hidden=64, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_cls),          # 输出 logits，不接 softmax
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
# 二分类标准配套：logits + BCEWithLogitsLoss（内部融合 sigmoid，数值稳定）
criterion = nn.BCEWithLogitsLoss()   # 多分类换 nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

for x, y in dataloader:               # x: (B, d_in), y: (B,) 或 (B, 1)
    logits = model(x).squeeze(-1)
    loss = criterion(logits, y.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

三个工程细节：①**输出层不接 softmax/sigmoid**，让损失函数内部融合（log-sum-exp 技巧，第 3 周展开）；②`BCEWithLogitsLoss` ≠ `BCELoss(sigmoid(x))`，前者数值稳定，永远用前者；③`optimizer.zero_grad()` 忘写 = 梯度累积，loss 会异常翻倍上涨——新手最常见 bug 之一。

### 深入：参数量计算（速算练习）

d_in=1000 → 64 → 64 → 2 的三层 MLP：

- W₁: 64×1000 = 64,000，b₁: 64
- W₂: 64×64 = 4,096，b₂: 64
- W₃: 2×64 = 128，b₃: 2
- **合计 ≈ 68K 参数**（FP32 约 272KB）

对比：全连接处理 224×224 图像的输入层一个矩阵就是 224²·C·n 量级——这就是为什么 CV 必须用卷积的权重共享（第 7 周）。

### 常见误区与自测要点

1. **"深 = 好"**——深的前提是每层有非线性 + 训练得动（初始化/归一化，第 6 周）；
2. **为什么隐藏层不用 softmax**：softmax 各维竞争和为 1，隐藏层需要各维独立表达不同特征；
3. **常见追问**：ReLU 为什么比 sigmoid 好？（梯度不饱和 + 计算便宜 + 稀疏激活）GELU 比 ReLU 好在哪？（平滑、门控式、经验上 Transformer 更稳）

### 延伸阅读

- 《神经网络与深度学习》（邱锡鹏）第 4 章
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*（2010，Xavier 初始化原文）
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)* (2016)

---

## 第 3 周 · 反向传播与训练流程

### 核心概念（速览）

**一句话**：反向传播 = 链式法则在计算图上的高效执行，一次前向 + 一次反向算出所有参数的梯度，代价仅约 2 倍前向计算。

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\cdot\frac{\partial z}{\partial x} \qquad\qquad \theta \leftarrow \theta - \eta \nabla_\theta L$$

**直觉类比**：训练像多人接力出错后追责：从最后一棒开始，每人按"自己对结果的影响程度"分责任（梯度），各自调整。链式法则保证追责可传递；反向传播保证只需走一遍流程。

**关键要点**：①前向存激活值，反向复用——显存 ≈ 参数量 + 激活值（后者随 batch size 线性增长）；②mini-batch 是噪声与成本的平衡点；③端到端流程任何一环出错都有标准排查路径；④数值稳定三大坑：exp 溢出、log(0)、除零。

### 深入：自动微分的两种模式

对 f: ℝⁿ → ℝᵐ：

- **前向模式（forward mode）**：沿计算图正向算"对某个输入方向的导数"，算一次得一行（一个输入对一个输出的敏感度）。适合 m ≫ n（输出多于输入），如敏感度分析；
- **反向模式（reverse mode）**：前向存中间值，反向从输出回溯，**一次反向得到所有输入对输出的导数**。神经网络 m=1（标量损失）、n=10⁹（参数），反向模式的性价比碾压，这就是"backprop"名字的由来。

**代价分析**：前向 FLOPs ≈ C，反向 ≈ 2C（每个基本算子的 VJP 约是其正变换的 2 倍），总计 ≈ 3C——"反向传播让训练成本仅 3 倍于推理"的更精确说法。

### 推导：log-sum-exp 技巧——softmax 与交叉熵为什么必须融合实现

朴素 softmax：pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}。若 z 中有 100，e^{100} ≈ 2.7×10⁴³ 直接溢出。

技巧：softmax(z) = softmax(z − c)，取 c = maxⱼ zⱼ，则最大指数项为 e⁰=1，其余 <1，**不可能溢出**。

同理 log pᵢ = zᵢ − c − log Σⱼ e^{zⱼ−c}。PyTorch 的 `F.cross_entropy` / `BCEWithLogitsLoss` 内部就是这么做的，所以**永远传 logits，永远不要在损失函数外面自己接 softmax/sigmoid**。

### 代码实践：标准训练循环（含验证与早停骨架）

```python
best_val, patience, wait = float('inf'), 5, 0
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防爆
        optimizer.step()

    model.eval()                       # 关键：关闭 dropout/BN 的训练行为
    val_loss = sum(criterion(model(x), y).item() for x, y in val_loader) / len(val_loader)
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        wait += 1
        if wait >= patience: break     # 早停
```

`model.train()` / `model.eval()` 忘记切换是高频事故：BN 用 batch 统计量 vs 滑动平均、Dropout 开与关，eval 模式下全不同——表现为"验证集指标莫名崩"或"保存的模型上线效果差"。

### 工程排查手册（loss 异常的标准决策流）

| 症状 | 最可能原因 | 排查动作 |
|---|---|---|
| loss = NaN（第 1 个 step） | 学习率过大 / 输入未归一化 / 自己实现了不稳定的 softmax | 降 lr 10 倍试跑；检查输入均值方差；换框架原生损失 |
| loss 不降 | 标签接错（y 全 0 / shuffle 错位）/ 学习率太小 / 输出层与损失不配套 | 抽查一个 batch 的 (x, y)；打印梯度范数；检查输出层 |
| 训练 loss 降、验证不降 | 过拟合 | 加数据/正则化/早停（第 6 周） |
| 训练 loss 不降、验证也不降 | 欠拟合（优化失败） | 加容量、换激活、查初始化与学习率 |
| loss 周期性震荡 | batch 太小 + lr 太大 | 降 lr 或 warmup（第 6 周） |

**梯度范数监控**是最强的通用诊断信号：`grad_norm = ‖∇L‖₂` 画曲线，正常应平稳下降；突然尖峰 = 遇到病态 batch，尖峰后 loss 爆炸 = 需要梯度裁剪；长期 ≈ 0 = 梯度消失（该查激活函数与初始化了）。

### 工程场景（原文案例复盘）

某审核模型上线后指标莫名劣化 → 训练 pipeline 里特征与标签错行对齐 → 模型认真学习了错误的映射。

**复盘要点**：深度模型没有任何"自觉性"，它会用全部容量去拟合你给它的任何映射，包括错的。防御手段：①训练前**抽样人工核对**若干 (x, y) 对；②训练前期观察 loss 是否降到"合理下限"以下（错标签的任务，loss 下界会异常低或异常高，视错位方式而定）；③上线前 shadow 对比新旧模型输出分布。

### 常见误区与自测要点

1. **"反向传播是一种优化算法"**——错，它是**算梯度的方法**；梯度下降/Adam 才是用梯度更新的优化算法。高频易错点；
2. **"梯度消失是因为链式法则连乘小于 1 的数"**——基本正确但要看激活函数：sigmoid 导数 ≤ 0.25 是元凶，ReLU 导数为 1 天然缓解（第 6 周 BN 进一步解决）；
3. **为什么显存随 batch 线性增长**：反向要复用每个中间激活值 a⁽ˡ⁾ 算 δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ，激活存储量 ≈ batch × Σ各层特征图大小。这也是梯度检查点（gradient checkpointing）"用计算换显存"的原理：重算一遍前向，只存少量节点。

### 延伸阅读

- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (2018)
- PyTorch 官方教程：AUTograd 机制、自定义 autograd.Function
- 《深度学习》（Goodfellow 等）第 6.5 节（反向传播的历史与代数结构）
