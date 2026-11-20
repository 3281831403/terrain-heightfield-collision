# terrain-heightfield-collision

## 题面

角色走上斜坡后，脚下发飘、贴地还会陷进地里半米：双线性插值按格心取整，斜面呈台阶；越界不 clamp，索引错误；法线用 `(h(x+1)-h(x), 1, h(z+1)-h(z))` 不归一，光照强度错；坡度用高度差而不是距离，格距变化时判错；射线只步进不细化，命中点偏半格；球碰撞只沿 Y 轴修正，斜坡上滑不下去；线段采样不按弧长，陡坡采样点过密；校验不查非矩形，索引越界；批量共用采样缓存，物体之间串；`spacing<=0` 不校验，除零。field/collision.py 是核心，tests/test_collision.py 15个红。要修的是：上面这 10 处缺陷，逐处修好且不改 tests，让 15 个用例全绿。

## 运行方式

```bash
cd terrain-heightfield-collision
python -m pytest tests -q
```

初始状态（未修复）：`15 failed`，没有 collection error / import error / syntax error——每个失败都是桩函数抛出的 `NotImplementedError`。修复后应得到 `15 passed`。

环境：Python 3 + pytest，离线可跑，只用标准库 `math`、`bisect`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约定

- 世界单位统一为米；高度场为 `{"origin": [x, z], "spacing": float, "heights": [[...], ...]}`，采样点 `heights[iz][ix]` 位于 `(origin[0] + ix * spacing, origin[1] + iz * spacing)`。
- 数值容差 `1e-6`（`field.collision.TOL`）；非法输入抛 `TerrainError`。
- `tests/` 不可修改：只允许改 `field/` 下的实现，测试文件与断言保持原样。
