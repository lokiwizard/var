# VAR 简介

本仓库实现的是 VAR（Visual Autoregressive Modeling）图像生成算法。它可以理解为“图像版 GPT”：先用 VQ-VAE 把图像压缩成离散 token，再用 Transformer 从粗到细预测这些 token，最后由 VQ-VAE decoder 解码回图像。

VAR 的关键不是逐像素生成，也不是按 raster 顺序一个 token 一个 token 地生成，而是按尺度生成：先生成 `1x1` 的粗 token map，再生成 `2x2`、`3x3`，一直到最高分辨率的 token map。粗尺度决定整体结构，细尺度补充局部纹理。

## 整体流程

1. **VQ-VAE 编码图像**  
   图像先经过 VQ-VAE encoder 得到连续潜变量 `f_BChw`。

2. **多尺度残差量化**  
   量化器从小尺度到大尺度依次处理潜变量残差，得到多组离散 token，例如 `1x1 -> 2x2 -> ... -> 16x16`。

3. **VAR 学习预测 token 金字塔**  
   Transformer 根据类别条件和已有尺度的信息，预测下一个尺度的 codebook id。

4. **自回归采样并解码**  
   推理时，VAR 从粗到细采样 token，逐步累加出完整潜变量 `f_hat`，最后交给 VQ-VAE decoder 还原成图像。

## 多尺度 VQ-VAE 如何工作

普通 VQ-VAE 通常只在一个固定分辨率上量化 latent。VAR 使用的是多尺度残差量化，核心变量有两个：

- `f_hat`：当前已经重建出的 latent。
- `f_rest`：还没有被解释掉的残差，初始等于 encoder 输出的 latent。

量化过程按尺度从小到大进行：

1. 将当前残差 `f_rest` 下采样到当前尺度，比如 `1x1`、`2x2`、`3x3`。
2. 对每个位置查找最近的 codebook 向量，得到该尺度的离散 token id。
3. 把查到的 codebook embedding 上采样回最大 latent 尺寸。
4. 经过一个小的残差卷积 `phi` 修正后，加到 `f_hat`。
5. 从 `f_rest` 中减去这部分贡献，让后续更细尺度只学习剩余细节。

这样，粗尺度 token 会优先表达全局结构，细尺度 token 主要表达高频细节。最终所有尺度的贡献相加，得到完整的量化 latent。

## 多尺度 VQ-VAE 如何训练

VQ-VAE 的训练目标是让量化后的 latent 仍然能重建原图。训练时流程如下：

1. 输入图像 `x` 经过 encoder 和 `quant_conv` 得到连续 latent。
2. 多尺度量化器从粗到细构造 `f_hat`，同时记录每个尺度使用的 codebook id。
3. `f_hat` 经过 `post_quant_conv` 和 decoder 得到重建图像。
4. 用重建图像和原图计算重建损失，同时量化器内部计算 VQ 损失。

量化器里的 VQ 损失包含两部分：

- codebook 需要靠近 encoder 输出的 latent。
- encoder 输出需要承诺到选中的 codebook embedding。

代码中使用 straight-through estimator：前向传播使用量化后的 `f_hat`，反向传播时让梯度可以传回 encoder。这样既能使用离散 token，又能端到端训练 encoder、decoder 和 codebook。

需要注意：当前仓库的 `train.py` 训练的是 VAR，并不会重新训练 VQ-VAE。代码会下载并加载预训练的 `vae_ch160v4096z32.pth`，然后冻结 VQ-VAE 参数。VQ-VAE 的前向和量化逻辑主要在 `models/vqvae.py` 和 `models/quant.py` 中。

## VAR 如何训练

VAR 训练时，VQ-VAE 是冻结的，只负责把图像变成训练目标 token。

对每个训练 batch：

1. 输入图像先经过冻结的 VQ-VAE，得到多尺度 token：  
   `gt_idx_Bl = [idx_1x1, idx_2x2, ..., idx_16x16]`

2. 把所有尺度的 token 拼成一个长序列，作为 VAR 的预测目标：  
   `gt_BL = concat(gt_idx_Bl)`

3. 构造 teacher forcing 输入。  
   第一个尺度没有图像 token 输入，只使用类别 embedding 和起始位置 embedding。后续每个尺度的输入不是该尺度自己的 token，而是由更早尺度重建出的累计 latent，再下采样到当前尺度。

4. Transformer 使用尺度级 causal mask：  
   当前尺度可以看到自己和更早尺度，不能看到未来尺度。同一尺度内部 token 可以互相看到，因为 VAR 是“按尺度并行预测”，不是逐 token 预测。

5. 输出每个位置上的 codebook 分类 logits，并和 `gt_BL` 做交叉熵损失。

训练目标可以简单理解为：

```text
给定类别条件 + 已有粗尺度信息，预测当前及后续尺度的真实 codebook id。
```

训练中还使用了 classifier-free guidance 的准备方式：以一定概率把真实类别替换成一个“无条件类别 id”。这样推理时可以同时跑条件分支和无条件分支，再用 CFG 公式增强类别条件。

## 第一阶段：训练多尺度 VQ-VAE

当前仓库新增了 `train_vae.py`，默认使用 `dataset/imagenet-10` 下的数据集。数据目录结构类似：

```text
dataset/imagenet-10/
  n02056570/
    xxx.JPEG
  n02085936/
    xxx.JPEG
  ...
```

单卡训练可以直接运行：

```bash
python3 train_vae.py \
  --data-path dataset/imagenet-10 \
  --batch-size 32 \
  --epochs 100 \
  --img-size 256 \
  --hflip \
  --vis-every 1 \
  --vis-scales 0,1,3,6,-1 \
  --amp
```

如果显存不够，优先减小 `--batch-size`。训练输出默认保存在 `checkpoints/vqvae/`：

- `vae-ckpt-last.pth`：每个 epoch 结束后覆盖保存的最新 checkpoint。
- `vae-ckpt-epXXX.pth`：按 `--save-every` 周期保存的 checkpoint。
- `vae_ch160v4096z32_custom.pth`：训练结束后保存的最终 checkpoint。
- `recon_epXXX.png`：按 `--vis-every` 保存的多尺度重建可视化图。

可视化图每一行对应一张样本，列的含义是：

```text
原图 | 指定尺度 1 的重建 | 指定尺度 2 的重建 | ... | 最终尺度重建
```

相关参数：

- `--vis-every 1`：每隔多少个 epoch 保存一次重建效果；设为 `0` 可关闭。
- `--vis-num 8`：每次可视化多少张样本。
- `--vis-scales 0,1,3,6,-1`：选择哪些尺度做可视化；`-1` 表示最后一个尺度，也可以设置为 `all` 输出全部尺度。

这个训练入口使用的是基础 VQ-VAE 目标：

```text
loss = 重建损失 + VQ 损失
```

其中重建损失默认是 L1，VQ 损失来自多尺度残差量化器。它可以训练出可用的 tokenizer，但如果追求官方级别的重建质量，通常还需要加入 perceptual loss 和 GAN discriminator loss。

## VAR 如何推理

推理时没有 ground truth token，模型完全自回归生成：

1. 从类别 embedding 和起始位置开始，预测最粗尺度 token。
2. 把采样出的 token 转成 codebook embedding，并累加到 `f_hat`。
3. 将当前累计 `f_hat` 下采样，作为下一尺度输入。
4. 重复直到最高分辨率尺度生成完成。
5. 用 VQ-VAE decoder 将最终 `f_hat` 解码成图像。

推理时会启用 KV cache，因此每个尺度只需要追加新的 K/V，不必反复计算历史尺度。

## 代码结构

- `models/vqvae.py`：VQ-VAE 编码器、解码器和量化入口。
- `models/quant.py`：多尺度残差向量量化，将图像 latent 转成 token 金字塔。
- `models/var.py`：VAR Transformer 主体，包含训练前向和自回归推理。
- `models/basic_var.py`：AdaLN、自注意力和 FFN 等 Transformer 基础模块。
- `trainer.py`：训练 step、验证、日志和 checkpoint 状态管理。
- `train.py`：训练入口，负责构建数据、模型、优化器和训练循环。
