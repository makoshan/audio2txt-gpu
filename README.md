# 🎙️ audio2txt-gpu

> 免费 GPU 音频转文字 · faster-whisper + Google Colab / AWS Studio Lab

## 快速开始（3 步）

### 方法一：Colab 直接打开 ⭐ 推荐

点击下方按钮，自动在 Colab 打开 notebook：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/makoshan/audio2txt-gpu/blob/main/audio_transcription_faster_whisper.ipynb)

1. 运行时 → **更改运行时类型 → GPU (T4)**
2. 把音频文件放到 **Google Drive 根目录**
3. 修改 `AUDIO_PATH`，全部运行

### 方法二：本地命令行

```bash
pip install faster-whisper
python transcribe.py audio.mp3
python transcribe.py audio.mp3 --model turbo   # 速度优先
python transcribe.py audio.mp3 --model large-v3 # 精度优先
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `audio_transcription_faster_whisper.ipynb` | Colab/Studio Lab notebook，含 GPU 检测、Drive 挂载、SRT 输出、WhisperX 扩展 |
| `transcribe.py` | 本地命令行脚本，Colab/Studio Lab/本地三环境通用 |
| `audio_transcription_report.html` | 可视化调研报告（方案对比、平台评级、决策树） |

## 模型选择

| 模型 | VRAM | 速度 | 中文精度 | 推荐场景 |
|------|------|------|---------|---------|
| `turbo` | ~6GB | ⚡ 快 | 好 | 速度优先 |
| `large-v3` | ~10GB | 稳 | ⭐ 最佳 | 精度优先 |

## 平台切换

Colab 额度用完时，切换到 [AWS Studio Lab](https://studiolab.sagemaker.aws/)：
- 改 `AUDIO_PATH = '/home/studio-lab-user/audio.mp3'`
- 其余代码**零改动**，环境和包持久化

## 需要说话人分离？

启用 notebook 中的 **Step 6 WhisperX** 部分，申请 [Hugging Face token](https://huggingface.co/settings/tokens) 后填入即可。

---

调研日期：2026-03-31
