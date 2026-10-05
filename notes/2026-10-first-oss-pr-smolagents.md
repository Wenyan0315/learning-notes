# 第一个 OSS PR：从发现 issue 到提交（smolagents `@` 运算符修复）

> 实战笔记 · 2026-10-05 · 状态：PR open，等待 review
>
> Issue：[huggingface/smolagents#2877](https://github.com/huggingface/smolagents/issues/2877)
> PR：[huggingface/smolagents#2901](https://github.com/huggingface/smolagents/pull/2901)

## 为什么做这个

12 周学习计划解决"系统补基础"，但读别人的代码和自己写代码之间还缺一环：**在真实的大型开源项目里改代码**。选 smolagents 的原因：

- 代码量小，核心就几个 Python 文件，读得完；
- 它是 agent 框架，和我从零写的 facta 同构（ReAct 循环、工具调用、代码执行器），等于给自己的项目找参考实现；
- 社区活跃，issue 有真实反馈。

## 全流程时间线

| 步骤 | 动作 | 说明 |
|---|---|---|
| 1 | 扫 issue | 用 GitHub API 按 `label:bug, state:open` 过滤，优先挑**小而清晰**的：报错明确、复现路径短、改动范围可预估 |
| 2 | 本地复现 | 确认是真 bug 而不是用法错误，这是提 PR 的前置条件 |
| 3 | issue 下留言认领 | 说明根因 + 修复计划。防止和别人撞车，也是让 maintainer 有预期 |
| 4 | fork + 分支 | `fix-matmul-operator`，一个 PR 只做一件事 |
| 5 | 修复 + 测试 | 仿照仓库现有测试风格写，不引入新框架 |
| 6 | 开 PR | `Fixes #2877` 关联 issue，What / Why / Testing 三段式 |

## Bug 解剖：AST 解释器的"白名单分发"

smolagents 执行 LLM 生成的代码时不用 `eval()`，而是 parse 成 AST 后**逐节点自解释**。二元运算在 `evaluate_binop()` 里长这样：

```python
if isinstance(binop.op, ast.Add):   return left_val + right_val
elif isinstance(binop.op, ast.Sub): return left_val - right_val
# ... Mult / Div / Mod / Pow / FloorDiv / BitAnd / ... / LShift / RShift
else:
    raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")
```

**问题**：白名单里有 12 个运算符，唯独没有 `ast.MatMult`（`@`）。agent 生成的任何 numpy 线性代数代码（`A @ B`、`X @ W.T`）直接抛 `NotImplementedError`。

**修复**：两个函数各加一个分支，共 4 行：

```python
# evaluate_binop()
elif isinstance(binop.op, ast.MatMult):
    return left_val @ right_val

# evaluate_augassign()
elif isinstance(expression.op, ast.MatMult):
    current_value @= value_to_add
```

注意 issue 标题写的是 "`@` **and** `@=`"——`@=` 走 `evaluate_augassign()`，是另一套分支，两处都要修。只修一处测试会挂在 `@=` 上。

## 测试怎么写

仿照仓库 `test_local_python_executor.py` 现有风格，numpy 数组断言结果值：

```python
def test_evaluate_binop_matmult(self):
    code = dedent("""\
        import numpy as np
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        A @ B
    """)
    state = {}
    result, _ = evaluate_python_code(code, {}, state=state)
    assert result.tolist() == [[19, 22], [43, 50]]
```

另外加了一个**负向冒烟**：`float @ int` 应该失败，但失败原因必须是 Python 语义的 `TypeError`，而不是 executor 白名单缺失的 `NotImplementedError`——以此区分"修好了"和"换了个报错"。

## 经验沉淀

1. **label 扫不到不代表没机会**。`good first issue` 被清空了，但 open 的 bug 列表里有的是小而清晰的切入点。新手选 issue 的标准：报错明确 > 需要理解架构 > 需要讨论设计。
2. **认领评论先行**。一行 "I'd like to work on this" + 根因分析，比直接甩 PR 更容易被接受，也避免白做。
3. **每个运算符一个分支的写法，漏一个就是运行时炸**。读这段代码时自然想到：如果写成 `{ast.MatMult: operator.matmul}` 映射表，就不会漏——白名单和映射表各有取舍，但"枚举式分发"的维护成本是真实的。这个观察已记进 facta 的设计备忘。
4. **功能白名单 ≠ 安全边界**。源码注释自己写明 "It is not a security sandbox"——它控制的是"能用什么功能"，不是"防恶意代码"。这两个概念在 agent 代码执行器的设计里经常被混。
5. **网络装不动依赖时别硬等**。给 `huggingface_hub` / `rich` 打桩后直接驱动真实执行器代码路径，照样能完成修复验证——环境问题不阻塞正确性验证。

## 下一步

- 等 review 反馈（已设定时监控，CI / 评论 / 合并都会提醒）
- 如果顺利 merge，把这段经历写进 facta README 的 Currently 区块
- 12 周计划照常推进，这篇作为"实战系列"第一篇
