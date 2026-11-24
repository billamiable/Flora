# Flora｜技术理解与实测

Flora 是 NVIDIA GEAR Lab 的机器人视频标注与评分工具。它从人类第一视角和机器人操作视频生成带时间戳的英文动作指令，并提供整段、过程和自定义规则评分。这些结果可用于训练数据筛选和策略评估。模型卡列出的发布日期为 **2026-10-15**。公开仓库提供推理示例，公开权重适用 NVIDIA 非商业研究许可。 [项目页](https://research.nvidia.com/labs/gear/flora/) · [论文](https://arxiv.org/abs/2610.31779) · [推理仓库](https://github.com/NVlabs/Flora#models-and-code) · [权重许可](https://huggingface.co/nvidia/Flora-TempSeg#licenseterms-of-use)

## 1. 整体流程

- **动作标注。** TempSeg 预测各窗口内的动作起止时间。后处理换算为原视频时间，并拼接跨窗的同一动作。Captioner 为每个区间生成描述，最终输出 `start_s`、`end_s` 和 `caption`。
- **视频评分。** Rewarder 独立运行，输入视频和任务说明。Rubric 模式还要提供自定义评分标准。
- **训练数据规模。** 三个模型各使用少于 **1 万小时**的视频。TempSeg 和 Captioner 各少于 **1,000 万条**样本，Rewarder 少于 **100 万条**。模型卡仅披露上限。 [论文 §4.3、附录 F/G](https://arxiv.org/abs/2610.31779) · [TempSeg 数据](https://huggingface.co/nvidia/Flora-TempSeg#training-dataset) · [Captioner 数据](https://huggingface.co/nvidia/Flora-Captioner#training-dataset) · [Rewarder 数据](https://huggingface.co/nvidia/Flora-Rewarder#training-dataset)
- **视频输入。** 公开标注命令用 `human` 处理人类单视角视频，用 `yam`、`dex` 或 `tianji` 处理机器人视频。机器人可用主相机单视角，也可用主相机和同步双腕相机。三相机画面按同一时刻拼图。公开动作标注脚本的单个输入视频上限为 120 秒。
- **动作标注成本。** 每小时源视频的 GPU 成本 = 运行小时数 × GPU 张数 × 每卡时单价 ÷ 源视频总小时数。论文按 8 张 GB300、每卡时 10 美元、约 1.235 条视频/秒估算，50 条视频（合计 1.245 小时）约需 40.5 秒，折合 **0.72 美元/源视频小时**。这个估算覆盖切分和描述。 [论文附录 D.2，公式（9）](https://research.nvidia.com/labs/gear/flora/assets/paper/flora.pdf#page=22)

## 2. 三个模型的输入、输出与示例

### 动作标注｜TempSeg 与 Captioner

| 模型 | 输入与采样 | 输出与后处理 | 示例 |
| --- | --- | --- | --- |
| [TempSeg](https://huggingface.co/nvidia/Flora-TempSeg) | 视频窗口和切分提示词。默认每窗 **16 秒、64 帧**（4 FPS），含中间 12 秒主标注区和两侧各 2 秒上下文，步长 12 秒。三视角 YAM 可用 `--yam32` 切换到 32 秒窗口。 | 输出窗口内的动作起止时间（0.05 秒刻度）和 `lo/ro` 延续标记。后处理换算到原视频时间，并拼接跨窗的同一动作。 | 左窗报 9–14 秒、右窗报 10–17 秒，配对后得到 **9–17 秒**（原视频时间）。 |
| [Captioner](https://huggingface.co/nvidia/Flora-Captioner) | 输入目标区间和前后画面，按 8 FPS 估算取帧数。目标区间最多 **40 帧**，前后文各最多 **12 帧**，合计最多 **64 帧**。Human 单视角和机器人三视角均在目标前后各取最多 2 秒画面。 | 每个目标区间生成一句英文指令，沿用原来的起止时间。 | 8 秒目标区间最多取 40 帧，覆盖整段。示例指令为 “Pick up the purple die and place it on the white plate.” |

### 视频评分｜Rewarder

**示例任务。** 将粉色碗里的香蕉和红色小物件倒入接收碗，再把粉色碗放回工作台。三种评分模式都接收视频和任务说明。 [Rewarder 模型卡](https://huggingface.co/nvidia/Flora-Rewarder)

| 模式 | 采样方式 | 输出与后处理 | 示例结果 |
| --- | --- | --- | --- |
| **Outcome** | 整段视频均匀采样，按 10 FPS 估算取帧数，最多 64 帧。 | 整段完成度（1–5 分）。 | **5 分**。 |
| **Dense** | 按 16 秒滑窗采样，每窗以 4 FPS 取 64 帧。默认步长 10 秒，末窗位置根据视频末尾调整。 | 每 0.5 秒输出一个过程质量分，再对齐到原视频时间。 | 若首次夹取时物体滑落，该时段可被评为 **1 分**。若重试后成功完成任务，整段 Outcome 仍可得 **5 分**。 |
| **Rubric** | 每路相机独立对整段视频采样，默认按 2 FPS 估算取帧数，每路最多 166 帧。 | 按自定义评分规则，默认每条视频生成 9 份候选答案。从可解析的答案中，选出与其余答案的逐项绝对分差之和最小的一份，输出逐项分和总分。 | 香蕉入碗 0.60、红色小物件入碗 0.20、粉色碗放回工作台 0.20，**总分 1.00**。 |

## 3. 实测结果

**实测配置。** H20 和 RTX PRO 5000 Blackwell 72 GB 各用一张卡，运行同一条 29.87 秒的 Human‑1M 笔记本操作视频。三个模型均逐条提交（batch size 1）。TempSeg 和 Captioner 用 PyTorch，Rewarder 用 vLLM（BF16、TP=1），Rubric 每个请求生成一份候选。

**复现样例。** 从打包的左侧主相机 MP4 提取完整 episode，即第 **3536–4431 帧**（从 0 开始，含首尾，共 896 帧）。 [Human‑1M 笔记本样例原视频（MP4）](https://huggingface.co/buckets/nvidia/human-1m-100k-oss/resolve/release-staging/oss_100k_v1/vendors/vendor_ee4c2f1e8a8c7b31e5903881ac5a000d/batch_bbef5102253455547184a1113e8eca3b/v0/shard_03/videos/observation.images.main_left/chunk-1674/file-249.mp4)

| 模型 | H20 整卡峰值（GiB） | RTX PRO 5000 整卡峰值（GiB） |
| --- | ---: | ---: |
| TempSeg | 19.19 | 21.29 |
| Captioner | 64.13 | 66.40 |
| Rewarder · Outcome / Dense | 84.95 | 65.96 |
| Rewarder · Rubric | 84.27 | 65.28 |

**为什么 H20 的 Rewarder 峰值更高。** 两张卡均按可见显存的 88% 配置 vLLM 预算。以 Outcome 为例，模型加载均占 51.01 GiB，H20 和 Pro 5000 的 KV cache 池分别为 29.48 和 8.39 GiB。整块预分配的缓存池都会计入整卡占用。 [vLLM 显存测算](https://docs.vllm.ai/en/stable/api/vllm/v1/worker/gpu_worker/#vllm.v1.worker.gpu_worker.Worker.determine_available_memory)

**显存测量。** 用 `nvidia-smi` 每 100 毫秒采样整卡显存，记录峰值。完整标注的峰值出现在 Captioner 阶段。

**首轮耗时参考。** 同一条 29.87 秒视频，计入模型加载和初始化。完整标注覆盖 TempSeg 切分到 Captioner 生成描述，Rewarder 各模式独立计时。

| 阶段 | H20 首轮（秒） | RTX PRO 5000 首轮（秒） |
| --- | ---: | ---: |
| TempSeg | 116.76 | 77.08 |
| Captioner | 260.21 | 275.00 |
| **完整标注** | **389.29** | **362.15** |
| Rewarder · Outcome | 82.71 | 241.73 |
| Rewarder · Dense（4 窗逐一提交） | 94.59 | 119.58 |
| Rewarder · Rubric（1 份候选） | 134.69 | 428.70 |

**Batch size 2 验证。** 同一视频上，TempSeg 在单卡 Pro 5K 跑通，Captioner 在单卡 H20 跑通。Captioner 在单卡 Pro 5K 遇到较长输入时出现显存不足。

**Rubric 默认配置。** 公开脚本默认用两张卡做张量并行（TP=2），两张卡分担模型权重并协作处理同一请求。每条视频默认生成 9 份候选，上下文上限为 98,304 token。本表的实测使用单卡 TP=1、1 份候选和 24,576 token 的上下文上限。 [公开脚本默认参数](https://github.com/NVlabs/Flora/blob/d8b755ec9a990418b1f31c7f48c75341c7229ea6/rewarder/batch_utils/rubric_reward.py#L54)
