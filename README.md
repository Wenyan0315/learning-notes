# learning-notes

Weekly deep-dives into high-quality repos and AI/agent topics — learning in public.

笔记以中文为主，每周一篇：读 README → 跑通 demo → 画调用链 → 沉淀成笔记。
另有一个进行中的 [12 周 AI 系统学习计划](notes/ai-system-12-weeks/)（机器学习基础 → 深度学习 → 经典模型，体系化学习笔记持续更新）。

仓库内容采用 [CC BY-SA 4.0](LICENSE) 协议发布。

## Notes

| 周 | 主题 | 笔记 |
|---|---|---|
| 2026-W40 | karpathy/nanoGPT 精读 | [notes/2026-09-nanogpt-deep-dive.md](notes/2026-09-nanogpt-deep-dive.md) |
| 2026-W41 | 第一个 OSS PR 实战：smolagents `@` 运算符修复（issue → 认领 → PR 全流程） | [notes/2026-10-first-oss-pr-smolagents.md](notes/2026-10-first-oss-pr-smolagents.md) |
| 进行中 | AI 系统学习笔记（12 周计划，第 1–39 天） | [notes/ai-system-12-weeks/](notes/ai-system-12-weeks/) |

## Articles

公众号「Wenyan的呜哇」文章归档——每周精读一篇 AI agent 方向论文。

| 日期 | 文章 |
|---|---|
| 2026-09-25 | [vLLM：用分页思想管好 KV Cache，推理吞吐提升 2–4 倍](articles/2026-09-vllm-pagedattention.md) |
| 2026-09-25 | [Agent 的"决定"，有 20.4% 在执行时悄悄变了样（confidence routing 论文精读）](articles/2026-09-confidence-routing-drift.md) |
| 2026-09-19 | [Agent 安全，从"守大门"到"守管道"：一份分层防御清单](articles/2026-09-agent-security-pipeline.md) |
| 2026-09-12 | [记忆越攒越多，Agent 怎么反而越答越错（Fortunate Recall 论文精读）](articles/2026-09-fortunate-recall-memory.md) |
| 2026-09-05 | [多智能体 LLM 系统通信机制：从协议标准化到神经信号传递](articles/2026-09-mas-communication.md) |
| 2026-08-30 | [【论文导读】AgentFold：当 AI 学会像一支迭代团队那样工作](articles/2026-08-agentfold.md) |
| 2026-08-20 | [让 Agent 跑完复杂任务：从 Self-Reflection 到 Independent Verification](articles/2026-08-longhorizon-harness.md) |
| 2026-08-11 | [ICLR 2026 论文解读：知识图谱和 RAG，什么时候用，什么时候不用](articles/2026-08-graphrag-when-to-use.md) |

## Method

每篇精读走同一条流水线：

1. **读 README 和 issue 区** — 搞清楚它解决什么问题、设计哲学是什么
2. **跑通 quickstart** — 跑不起来的项目不值得精读
3. **挑一个核心模块画调用链路** — 从入口到出口，搞清楚数据怎么流
4. **写成笔记** — 用自己的话复述，能讲清楚才算读懂
