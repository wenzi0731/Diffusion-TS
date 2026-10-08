# Baseline-4：Conditional Diffusion-TS（HEEW 四通道条件适配版）

基于本仓库 Diffusion-TS 官方实现，保留 encoder–decoder Transformer、多项式趋势分支、
Fourier 季节分支、直接重建 x0、时间域 L1 + Fourier 实部/虚部 L1 和原版 timestep loss weighting。
新增代码放在 `baseline4/`，导入 `Models/interpretable_diffusion/` 原始模块。
上游版本：`566307e6cf2d8095e58de4c6e3a6ae965b69b5b5`，沿用仓库 MIT 许可证。

这是针对外生条件场景生成的适配，论文表格建议标为 **Diffusion-TS (conditional adaptation)**。
原论文的 forecasting/imputation 是对已观测序列的条件补全；本适配在训练和每个去噪步骤将
干净的目标日天气/时间特征嵌入加入目标序列 embedding，不使用原版 infill/Langevin 接口。
没有额外 Stage-1 点预测器，也不需要 OOF。

## 数据与模型接口

- 输入：目标日 24 小时 `pv10` 天气、8 个周期时间特征、连续年份 `(Year-2014)/6`。
- 连续年份作为共享条件 embedding 的一部分，与 baseline-3 的条件可见性一致；它不只作用于 PV 输出。
- 无历史能源条件；测试真实能源值仅用于评估，不输入采样器。
- 一次联合生成 `Electricity, Heat, Cooling, PV` 四通道，内部序列 `[B,24,4]`。
- 导出场景 `[day,scenario,channel,hour]`，与现有 baseline 一致。
- 按完整日划分：2014–2020 train，2021 validation，2022 test。
- 按时间戳对齐能源与天气，排除不完整日；归一化仅使用训练完整日统计量。
- 使用 Z-score，所以不使用原版 `[-1,1]` 采样截断。反归一化后仅将负 PV 截为 0，与 baseline-3 一致。
- 这套条件是给定天气条件下的场景生成；CSV 中的天气值不会自动转为天气预报。

本次对齐的是 GitHub `wenzi0731/wenzia` 中 `2-stages` 的 `pv10`、年份切分及现有 baseline-1/2/3 协议。
如果另一个本地实验使用不同数据集或切分，请先统一所有方法的实验协议。

## 1. 安装与数据

在仓库根目录运行（建议使用 Python 3.10+，CPU 或 CUDA）：

```bash
pip install -r requirements_baseline4.txt
```

使用独立依赖清单即可，不需要原仓库中用于 MuJoCo/其他实验的全部环境。
将已有数据放入下列位置，或编辑 `baseline4/configs/heew.yaml` 的两个路径：

```text
Data/CN03_energy_cleaned.csv
Data/weather_cleaned.csv
```

能源 CSV 必需列：`Year,Month,Day,Hour,Electricity,Heat,Cooling,PV`。
天气必需列：`Year,Month,Day,Hour,Temperature,Dew Point,Humidity,Wind Speed,Pressure,Precip,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,PV_CLEARNESS_RATIO,PV_IS_DAYLIGHT`。
数据、checkpoint、实验结果均不上传仓库。

所有相对路径均相对于仓库根目录解析。请使用 `python -m baseline4...`，不要在 `baseline4` 子目录直接运行文件。

## 2. 先执行严格六组调参

```bash
python -m baseline4.sweep \
  --config baseline4/configs/heew.yaml \
  --search-space baseline4/configs/search_space.yaml \
  --seed 42
```

| 配置 | d_model | 学习率 |
|---|---:|---:|
| c01_w96_lr1e5 | 96 | 1e-5 |
| c02_w96_lr5e5 | 96 | 5e-5 |
| c03_w96_lr1e4 | 96 | 1e-4 |
| c04_w128_lr1e5 | 128 | 1e-5 |
| c05_w128_lr5e5 | 128 | 5e-5 |
| c06_w128_lr1e4 | 128 | 1e-4 |

固定参数：encoder 4 层、decoder 3 层、4 heads、cosine schedule、训练扩散步数 T=1000、
DDIM 采样步数=100、eta=0、batch=32、最多300个完整训练 epoch、每5 epoch验证一次、
连续12次验证不改善早停、EMA decay=0.995（每次 optimizer update 后更新）。
这是本 baseline 的预声明训练协议，并非逐项复刻原仓库的 optimizer update 数或 EMA 更新频率。

验证使用20场景/天，按四通道 `val_macro_nCRPS` 最小值选择 EMA checkpoint 和配置。
每通道 nCRPS = 全部验证日小时 CRPS 之和 / 真实值绝对值之和；macro 是四通道等权平均。
测试集不会由搜索脚本评估。程序要求恰好六个不同配置，搜索字段限于宽度和学习率。
相同试验数不代表相同 FLOPs；宽度不同会改变参数量，训练时间和参数量另行保存。

输出目录：

```text
experiments/baseline4/tuning_budget_6/
  protocol.yaml
  partial_results.json
  tuning_results.csv
  best_config.json
  best_config.yaml
```

`tuning_results.csv` 从优到差排序，`best_config.yaml` 是自动导出的完整胜出配置，可直接训练。
已经完整结束的试验可通过 `--skip-completed` 跳过，程序检查完成标记及配置指纹。
中途失败的试验不自动续训；保留或移走该试验的不完整输出目录后，以同一配置/seed重跑。
已有非空训练目录会报错，防止混用旧 checkpoint。不要在观察测试结果后增加第七组配置。

## 3. 最佳配置跑五个正式种子

不需要手工从 JSON 抄写参数：

```bash
for seed in 42 123 777 2024 3407; do
  python -m baseline4.train \
    --config experiments/baseline4/tuning_budget_6/best_config.yaml \
    --run-name diffusionts_final \
    --set run.seed=$seed
done
```

每个种子的最好 EMA 模型在 `experiments/baseline4/diffusionts_final_seed42/best.pt` 等目录。
同时保存 resolved config、history、训练数据文件 SHA256、归一化参数、参数量及环境版本。
训练使用相同年份切分，不将2021验证集并入训练集。配置确定后无需为每个种子重新调参。

## 4. 测试与结果位置

```bash
python -m baseline4.evaluate \
  --checkpoint experiments/baseline4/diffusionts_final_seed42/best.pt \
  --scenarios 100 \
  --seed 42
```

默认输出仍在 checkpoint 同级的 `evaluation_test/`。更新后请使用新的空目录：

```bash
python -m baseline4.evaluate \
  --checkpoint experiments/baseline4/diffusionts_final_seed42/best.pt \
  --outdir evaluation_results/baseline4_seed42_aligned
```

本次仅对齐 baseline5（`17c217b`）的评估定义和输出，不修改模型、训练、EMA、
六组搜索、训练扩散步数、DDIM eta=0 采样、采样分块或随机数流程，无需重新训练。
数据 SHA256 和训练归一化统计的 checkpoint 校验保持不变；路径可改，文件内容须一致。
已有非空输出目录拒绝覆盖。`--output-dir` / `--num-scenarios` 分别是
`--outdir` / `--scenarios` 的别名；`--no-plots` 跳过绘图及 Pearson 对比指标。
`--max-days` 仅供排错，输出会标明 `partial_test`，不可作为完整测试集主表。

### 输出文件

```text
channel_metrics.csv
global_metrics.csv
global_metrics.json
global_metrics_wide.csv
metrics_summary.json
daily_scores.csv
metric_definitions.json
baseline4_scenarios.npz
global_pearson.png
pearson/real_global_pearson.png
pearson/generated_global_pearson.png
pearson/pearson_difference.png
random_timeseries_50/
```

- `channel_metrics.csv`：每通道一行，MAE、RMSE、MAE_Z、RMSE_Z、R2、CRPS、
  nCRPS、Precision_Z、Recall_Z、CR、IW、IS、IS_Z。
- `global_metrics.csv` 保留原 `metric,value` 两列；`global_metrics_wide.csv` 为额外的单行宽表。
- `global_metrics.json` / `metrics_summary.json` 包含汇总指标、ES/VS、checkpoint epoch、
  训练种子、评估种子、实际噪声种子、训练/采样步数、batch/chunk size 和采样耗时。
- `daily_scores.csv` 包含日期、逐日 ES_Z/VS_Z、每通道 CRPS、IS、IS_Z，可用于配对分析。
- `metric_definitions.json` 记录公式口径及 VS 配对范围。
- NPZ 保留原数组名，增加通道名称、PV 年份条件、seed、sample_seed、sampler 和 JSON 元数据。
  场景/真实目标是物理量；conditions 是标准化条件；target_mean/std 仅来自训练集。
- `global_pearson.png` 是真实数据全局相关矩阵，生成/差异矩阵在 `pearson/`。
  随机图选取不重复的最多 50 个测试日。

### 与 baseline5 一致的指标口径

1. 点预测使用场景均值。R² 在每通道全部测试日/小时上计算，不是 Pearson²；
   真实目标恒定时 R² 未定义，写为 JSON null / CSV 空单元格。
2. MAE_Z/RMSE_Z 使用训练集逐通道 Z-score，不是 min-max。nCRPS 是逐点经验 CRPS
   总和除以真实值绝对值总和加 1e-8。`macro_nCRPS` 为四通道等权均值；
   `mean_nCRPS` 保留合并通道后的比值，两者不同。
3. CR、IW、IS 使用 95% 中央区间、线性 2.5%/97.5% 分位数、alpha=0.05。
   IS_Z = 物理 IS / 训练通道标准差；IS95 是 IS 别名。
   旧 Coverage90/IntervalWidth90 另行保留，不混用置信水平。
4. ES/ES_Z 在标准化的 96 维日联合轨迹上计算，经验分布成对项分母为 S²。
5. VS/VS_Z 使用论文定义：p=0.5，跨通道无序配对权重 1、同通道权重 0，
   同时包含同小时和不同时小时，**3456 个配对先求和，再按天平均，不除以配对数**。
6. Precision/Recall 保留最近中心半径近似，不是所有 kNN 球并集的精确判定。
   默认 k=5、生成日曲线池上限 10000，抽样种子 seed+1000+channel。
7. 生成 PV 反归一化后负值截为 0，再使用同一批场景计算所有指标。

测试 seed 默认从 checkpoint 继承，初始噪声仍用 seed+90000；验证仍用 seed+50000。
训练 T、测试采样 steps 和 batch/chunk size 均沿用 checkpoint 配置。
DDIM eta=0 仍由初始噪声产生多场景。复现时须固定硬件、依赖版本、batch 和 chunk size。

### 旧 NPZ 直接补算，不训练、不重新采样

```bash
python -m baseline4.score_npz \
  --npz experiments/baseline4/diffusionts_final_seed42/evaluation_test/baseline4_scenarios.npz \
  --outdir evaluation_results/baseline4_seed42_rescored
```

旧 baseline4 NPZ 已有训练均值/标准差，但未保存 seed，请保留旁边原始
`global_metrics.json`，程序会自动读取其中的 seed/sample_seed。
JSON 若移到别处，用 `--metrics-json` 指定。若缺少种子来源，需显式提供
`--seed` 用于指标抽样和画图；该参数不会重新采样，也不改变原始生成种子。
目录名不能证明实际种子。

对于采用非默认 Precision/Recall 参数的旧结果，请额外提供匹配的 `--checkpoint`，
或用 `--precision-recall-k` / `--max-precision-samples` 指定原设置。
优先级：命令行 > NPZ/原指标 JSON > checkpoint 配置 > 默认 5/10000。
提供 checkpoint 时会校验其训练归一化统计和归档的一致性。
若自定义旧 NPZ 缺少训练统计量，可用匹配 checkpoint 的 `data_manifest` 恢复；
绝不使用测试集估计均值/标准差。

补算仅向新目录写表格和图，不修改或重复保存源 NPZ。
最终报告五次完整训练/测试的 mean ± std；六组试验不包含五种子的最终重复实验。

## 验证与引用

```bash
python -m pytest -q
```

测试覆盖原版输出/损失数值对齐、条件梯度、采样可重复性、时间切分及训练统计、六组约束、
合成数据完整搜索与最佳配置再训练、checkpoint加载、测试CSV/NPZ及绘图输出；
新增论文 ES/VS 公式、旧采样逐元素一致性、不同分块/部分测试、旧归档补算及种子恢复测试。
合成数据测试不是训练收敛或基准性能证明，完整六组实验需在你的训练环境运行。

引用原论文：Xinyu Yuan and Yan Qiao, “Diffusion-TS: Interpretable Diffusion for General Time Series Generation”, ICLR 2024.
论文链接：https://openreview.net/forum?id=4h1apFjO99
