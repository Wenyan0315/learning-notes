# learning-notes

公开学习笔记，以中文为主。这里收录斯坦福 CS336 课程精读、12 周 AI 基础笔记，以及开源项目与 Agent 论文精读。

想从机器学习和深度学习基础开始，可以沿着 [12 周 AI 系统学习计划](notes/ai-system-12-weeks/) 阅读，每章配有推导、代码和自测要点。

仓库内容采用 [CC BY-SA 4.0](LICENSE) 协议发布。

## Stanford CS336 精读

按讲次阅读，从语言模型的计算量估算一路到分布式训练和缩放律。

| 讲次 | 阅读入口 |
|---|---|
| 第 1–2 讲 | [从 Token 到 FLOPs](articles/2026-10-cs336-lecture-1-2.md) |
| 第 3–4 讲 | [从 RoPE 到 MoE](articles/2026-10-cs336-lecture-3-4.md) |
| 第 5–6 讲 | [从算力到带宽](articles/2026-11-cs336-lecture-5-6.md) |
| 第 7–8 讲 | [从一张卡到一个集群](articles/2026-11-cs336-lecture-7-8.md) |
| 第 9、11 讲 | [从幂律到算力最优](articles/2026-11-cs336-lecture-9-11.md) |

## Notes

| 周 | 主题 | 笔记 |
|---|---|---|
| 2026-W40 | karpathy/nanoGPT 精读 | [notes/2026-09-nanogpt-deep-dive.md](notes/2026-09-nanogpt-deep-dive.md) |
| 2026-W41 | 第一个 OSS PR 实战：smolagents `@` 运算符修复（issue → 认领 → PR 全流程） | [notes/2026-10-first-oss-pr-smolagents.md](notes/2026-10-first-oss-pr-smolagents.md) |

## Articles

公众号「Wenyan的呜哇」文章归档——AI agent 论文精读与 AI 系统主题长文。

| 日期 | 文章 |
|---|---|
| 2026-11-22 | [从幂律到算力最优：斯坦福 CS336 第 9、11 讲精读](articles/2026-11-cs336-lecture-9-11.md) |
| 2026-11-15 | [从一张卡到一个集群｜斯坦福 CS336 第 7–8 讲精读](articles/2026-11-cs336-lecture-7-8.md) |
| 2026-11-08 | [从算力到带宽：斯坦福 CS336 第 5–6 讲精读](articles/2026-11-cs336-lecture-5-6.md) |
| 2026-10-31 | [从 RoPE 到 MoE：斯坦福 CS336 第 3–4 讲精读](articles/2026-10-cs336-lecture-3-4.md) |
| 2026-10-26 | [从 Token 到 FLOPs：斯坦福 CS336 前两讲精读](articles/2026-10-cs336-lecture-1-2.md) |
| 2026-10-20 | [DeepSeek 开源 DeepInfer：推理提速，为什么要从模型设计开始？](articles/2026-10-deepinfer-inference.md) |
| 2026-10-17 | [一千个人同时提问，大模型是怎么扛住的？](articles/2026-10-llm-inference-serving.md) |
| 2026-10-11 | [本地复刻一个 Jev：把开源大模型变成极速决策引擎（无需重新训练）](articles/2026-10-jev-local-replica.md) |
| 2026-10-10 | [大模型为什么一个字一个字地"蹦"？四篇论文讲清推理加速](articles/2026-10-llm-inference-optimization.md) |
| 2026-10-02 | [生成归生成，判断归判断：不是范式革命，是判断层的回归与校准升级](articles/2026-10-decision-model-jev.md) |
| 2026-09-25 | [vLLM：用分页思想管好 KV Cache，推理吞吐提升 2–4 倍](articles/2026-09-vllm-pagedattention.md) |
| 2026-09-25 | [Agent 的"决定"，有 20.4% 在执行时悄悄变了样（confidence routing 论文精读）](articles/2026-09-confidence-routing-drift.md) |
| 2026-09-19 | [Agent 安全，从"守大门"到"守管道"：一份分层防御清单](articles/2026-09-agent-security-pipeline.md) |
| 2026-09-12 | [记忆越攒越多，Agent 怎么反而越答越错（Fortunate Recall 论文精读）](articles/2026-09-fortunate-recall-memory.md) |
| 2026-09-05 | [多智能体 LLM 系统通信机制：从协议标准化到神经信号传递](articles/2026-09-mas-communication.md) |
| 2026-08-30 | [【论文导读】AgentFold：当 AI 学会像一支迭代团队那样工作](articles/2026-08-agentfold.md) |
| 2026-08-20 | [让 Agent 跑完复杂任务：从 Self-Reflection 到 Independent Verification](articles/2026-08-longhorizon-harness.md) |
| 2026-08-11 | [ICLR 2026 论文解读：知识图谱和 RAG，什么时候用，什么时候不用](articles/2026-08-graphrag-when-to-use.md) |

## Method

开源项目精读按下面的步骤进行：

1. **读 README 和 issue 区** — 搞清楚它解决什么问题、设计哲学是什么
2. **跑通 quickstart** — 跑不起来的项目不值得精读
3. **挑一个核心模块画调用链路** — 从入口到出口，搞清楚数据怎么流
4. **写成笔记** — 用自己的话复述，能讲清楚才算读懂
