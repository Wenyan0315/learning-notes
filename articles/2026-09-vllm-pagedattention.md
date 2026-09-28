# vLLM：用分页思想管好 KV Cache，推理吞吐提升 2–4 倍

> 原文首发于微信公众号「Wenyan的呜哇」，2026-09-25。[阅读原文](https://mp.weixin.qq.com/s/_8SnwKQkz7fzPQEsV1FK1Q)

大模型推理的成本问题，可以归结为一句技术判断：在显存容量固定的前提下，KV Cache 的管理效率决定了系统能并行处理多少请求，进而决定整体吞吐。

2023 年，UC Berkeley 团队在 SOSP 发表《Efficient Memory Management for Large Language Model Serving with PagedAttention》，给出了解决方案。论文引用已破 9000，其系统实现 vLLM 成为今天开源界使用最广泛的 LLM 推理引擎。

论文与 vLLM 是同一项工作的两面。PagedAttention 算法与整套分页内存管理机制出自论文，vLLM 是这些机制的第一个完整实现——文中描述的调度器、块管理器和 CUDA kernel，构成了 vLLM 最初的代码本体。不过论文定格在 SOSP 2023 接收时的状态：当时只支持 OPT、LLaMA 等少数模型，全部代码约一万行；作为开源项目的 vLLM 此后快速演进，模型架构支持扩展到数百种，陆续加入了流水线并行、prefix caching、结构化输出、多模态等论文并未覆盖的能力。可以这么定位二者：论文给出的是 vLLM 的设计原则与起点，而今天在生产环境中运行的 vLLM，是这套骨架在数年社区演进后长成的完整系统。

这篇文章按由浅入深的顺序，把论文的问题定义、核心方法和实验结论讲清楚。

---

## 一、背景：LLM 推理的计算特征

LLM 采用自回归方式生成文本：每输出一个 token，都依赖此前全部上下文。生成过程分为两个阶段，计算特征截然不同。

预填充（Prefill）阶段：输入的 prompt 一次性进入模型，执行矩阵-矩阵乘法，计算密度高，GPU 利用充分。

解码（Decode）阶段：逐 token 串行生成，每次迭代只处理一个 token，执行矩阵-向量乘法。此时大部分耗时花在从显存读取模型权重和缓存上，属于典型的访存受限（memory-bound）场景，GPU 算力大量闲置。

提升利用率的标准手段是批处理（batching）：多个请求合并计算，摊薄权重读取开销。但 batch 规模受显存限制——其中 KV Cache 是显存的主要动态消耗。

KV Cache 的规模：以 130 亿参数的 OPT 模型为例，缓存一个 token 的 Key/Value 向量需要约 800 KB，2048 个 token 的请求即占用 1.6 GB。一张 A100 的显存中，模型权重约占 65%，KV Cache 预算通常在 30% 左右。权重部分固定不可压缩，因此 KV Cache 的管理水平直接决定可支持的并发请求数。

结论：推理优化的关键路径，是 KV Cache 的显存管理。

## 二、问题：既有方案的效率缺陷

当时的代表系统 FasterTransformer 和 Orca 存在一个共同约束：KV Cache 在显存中以连续张量形式存放。这一约束来自深度学习框架的通用设计，普通张量适用，但 KV Cache 有两个特殊性质——长度在请求开始前不可知，且随生成过程动态增长。为适配连续存储，系统只能按请求的最大序列长度静态预分配一整块显存。

论文将由此产生的浪费归纳为三类：

1. 预留（reserved）：为未来 token 预占的空间，在整个请求生命周期内不可被其他请求使用；
2. 内部碎片（internal fragmentation）：请求实际长度远小于最大长度，多分配的部分全程闲置；
3. 外部碎片（external fragmentation）：大小不一的分配块交错释放后，产生无法复用的内存空隙。

实测数据：既有系统中 KV Cache 显存的实际利用率仅为 20.4%–38.2%。

更关键的是，连续存储还阻断了共享。parallel sampling 与 beam search 等解码算法下，同一请求的多个输出序列天然共享 prompt 部分的 KV，但由于各序列被隔离在独立的连续空间中，这部分冗余存储无法消除。

## 三、核心方法：PagedAttention 与分页管理

作者注意到，这一问题与操作系统在 1962 年解决的内存管理问题在结构上完全同构。PagedAttention 即是这一观察的系统化落地，类比如下：

| 操作系统 | vLLM |
| --- | --- |
| 页（Page） | KV 块（默认 16 个 token） |
| 字节 | token |
| 进程 | 请求 |
| 页表 | 块表（Block Table） |

算法层：非连续注意力。注意力 kernel 原本假设 K、V 张量在显存中连续排布。PagedAttention 将 KV Cache 切分为固定大小的块，块在物理显存中可任意离散存放，kernel 通过块表定位各块的物理地址，按块完成打分、加权与归一化。该改写与原始注意力在数学上严格等价，即不损失任何精度。

管理层：按需分配。逻辑块自左向右顺序填充，物理块仅在上一块写满后分配。由此，预留与外部碎片被消除，内部碎片收敛到最后一个未写满的块——单个请求的浪费被限制在一个块以内，实现近零浪费。请求结束后其块立即释放，供其他请求复用。

## 四、机制扩展：从内存节省到系统能力

分页机制不仅解决浪费，还顺带打开了既有架构无法实现的几个能力。

块级共享与写时复制。对共享相同前缀的多个序列，其逻辑块可映射到同一物理块，并以引用计数跟踪共享者数量。当某个序列需要写入共享块时，触发写时复制（copy-on-write）：分配新块、复制内容、递减引用计数。beam search 中候选序列动态分化与淘汰，共享拓扑随迭代变化，均通过引用计数的增减自动管理，全程无大段数据拷贝。系统 prompt 的 KV 可预先缓存为公共物理块，新请求直接映射——即后续产品化中的 prefix caching。

显存压力下的调度。采用先来先服务调度保证公平；驱逐策略为 all-or-nothing——一条序列的 KV 块要么全部保留、要么全部移出，因为注意力计算会整体访问该序列的全部缓存。被驱逐序列恢复时有两条路径：将块换出至 CPU 内存（swapping），或将已生成 token 拼接回 prompt 一次性重建全部 KV Cache（recomputation）。后者因可并行执行，通常显著快于原始逐 token 生成。

分布式执行。支持 Megatron-LM 风格的张量并行。系统设单一集中式 KV Cache 管理器，所有 GPU worker 共享同一份逻辑块到物理块的映射，各 worker 仅存储其所负责 attention head 对应的 KV 分片。调度器每轮迭代广播一次控制信息，worker 间无需为内存管理进行额外同步。

## 五、工程实现

分页引入离散访存模式，通用 kernel 无法高效处理，团队做了三项定制：将 KV 切块、布局调整与按块表写入融合为单次 kernel 调用；改造注意力 kernel 为一个 warp 负责读取一个块，保证访存合并（coalesced access）；将写时复制产生的多次零散拷贝合并为单次调用。系统总计约 8500 行 Python（调度与控制）和 2000 行 C++/CUDA（kernel）。

## 六、实验结论

主结果：在 13B、66B、175B 三个规模的 OPT 模型上，与 FasterTransformer 和 Orca 对比，同延迟水平下 vLLM 吞吐提升 2–4 倍，且序列越长、模型越大、解码算法越复杂，优势越显著。

机制验证：监控数据显示 vLLM 可同时承载的请求数明显多于基线，"省显存 → 更大 batch → 更高吞吐"的因果链完整闭环。

消融结论：分页 kernel 因访存不连续，单算速度略低于基线，但被显存利用率的提升完全覆盖；块大小取 16 是延迟与碎片的平衡点；抢占恢复中 recomputation 在多数配置下优于 swapping。

## 七、评价与局限

方法论价值。分页、写时复制、共享缓存均为上世纪六十年代的成熟机制。本文的贡献不在于提出新技术，而在于准确识别出"KV Cache 管理"与"进程内存管理"的结构同构，并完成了严格对应的机制迁移——包括 all-or-nothing 驱逐这类基于注意力计算特性做出的合理改造。

零代价收益。与量化（损失精度）、蒸馏（损失能力）不同，本方法不改变任何模型输出，纯系统层优化即带来数倍吞吐。

历史影响。"调度器 + 块管理器 + 分页注意力 kernel"的架构被 TensorRT-LLM、SGLang、LMDeploy 等后续框架广泛继承；prefix caching、continuous batching 等功能已成为推理服务的标准配置。

局限。PagedAttention 面向传统 MHA 结构设计；GQA、MLA 等后续架构从注意力机制本身压缩了 KV Cache 体积，Prefill/Decode 分离部署等新范式也在重构推理系统架构。分页机制在新一代技术格局中的演化方向，是值得继续关注的问题。

## 参考文献

1. Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. arXiv:2309.06180
2. Vaswani A, et al. Attention Is All You Need. NeurIPS 2017.
3. Yu G, et al. Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI 2022.
