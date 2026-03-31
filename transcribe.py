#!/usr/bin/env python3
"""
音频转文字 — faster-whisper 本地 / Colab / Studio Lab 通用脚本
调研日期: 2026-03-31

用法:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --model large-v3 --lang zh
    python transcribe.py audio.mp3 --model turbo --no-srt

依赖:
    pip install faster-whisper
    # 如需 .m4a/.aac 等格式支持（本地运行）：
    # brew install ffmpeg  (macOS)
    # apt install ffmpeg   (Linux)
"""

import argparse
import os
import sys
import time


def fmt_time(sec: float) -> str:
    """秒 → SRT 时间格式 HH:MM:SS,mmm"""
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def check_gpu():
    """检测运行环境并返回合适的 device / compute_type"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {name} ({vram:.1f} GB VRAM)")
            # VRAM < 8GB 用 int8 节省显存
            compute_type = "float16" if vram >= 8 else "int8_float16"
            return "cuda", compute_type
    except ImportError:
        pass
    print("⚠️  未检测到 GPU，回退到 CPU（速度慢，建议切换到 Colab/Studio Lab GPU 环境）")
    return "cpu", "int8"


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str | None = "zh",
    batch_size: int = 16,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    output_dir: str | None = None,
    write_srt: bool = True,
):
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    if not os.path.exists(audio_path):
        print(f"❌ 找不到音频文件: {audio_path}")
        sys.exit(1)

    file_mb = os.path.getsize(audio_path) / 1024 / 1024
    print(f"📂 音频文件: {audio_path}  ({file_mb:.1f} MB)")

    device, compute_type = check_gpu()

    # ---- 加载模型 ----
    print(f"\n⏳ 加载模型 {model_size} (device={device}, compute_type={compute_type}) ...")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    if device == "cuda":
        pipeline = BatchedInferencePipeline(model=model)
        segments_gen, info = pipeline.transcribe(
            audio_path,
            batch_size=batch_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            language=language,
        )
    else:
        # CPU 模式不使用 BatchedInferencePipeline（无收益）
        segments_gen, info = model.transcribe(
            audio_path,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            language=language,
        )

    print(f"✅ 模型加载耗时 {time.time()-t0:.1f}s")
    print(f"\n⏳ 开始转录...")
    t1 = time.time()

    segments = list(segments_gen)  # 展开 generator
    elapsed = time.time() - t1

    print(f"✅ 转录完成！共 {len(segments)} 段，耗时 {elapsed:.1f}s")
    print(f"   检测语言: {info.language}，置信度: {info.language_probability:.2%}")
    print(f"   音频时长: {info.duration:.1f}s  →  实时率: {info.duration/elapsed:.1f}×")

    # ---- 输出路径 ----
    base = os.path.splitext(audio_path)[0]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, os.path.basename(base))

    txt_path = base + "_transcript.txt"
    srt_path = base + "_transcript.srt"

    # ---- 纯文本 ----
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")
    print(f"\n📄 纯文本: {txt_path}")

    # ---- SRT 字幕 ----
    if write_srt:
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{fmt_time(seg.start)} --> {fmt_time(seg.end)}\n")
                f.write(seg.text.strip() + "\n\n")
        print(f"🎬 SRT 字幕: {srt_path}")

    # ---- 预览前 5 段 ----
    print("\n--- 前 5 段预览 ---")
    for seg in segments[:5]:
        print(f"  [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text.strip()}")

    return txt_path, srt_path if write_srt else None


def main():
    parser = argparse.ArgumentParser(
        description="音频转文字 (faster-whisper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python transcribe.py meeting.mp3
  python transcribe.py meeting.mp3 --model turbo --lang zh
  python transcribe.py meeting.mp3 --model large-v3 --out-dir ./output
  python transcribe.py meeting.mp3 --no-srt

模型大小参考 (T4 GPU):
  tiny      ~1GB VRAM, 最快, 精度最低
  base      ~1GB VRAM
  small     ~2GB VRAM
  medium    ~5GB VRAM
  turbo     ~6GB VRAM, 速度/精度均衡 ⭐
  large-v3  ~10GB VRAM, 最慢, 中文精度最佳 ⭐

平台切换提示:
  Colab:       AUDIO_PATH = '/content/drive/MyDrive/audio.mp3'
  Studio Lab:  AUDIO_PATH = '/home/studio-lab-user/audio.mp3'
  本地:        直接传文件路径
        """,
    )
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--model", "-m", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "turbo",
                                 "large-v1", "large-v2", "large-v3"],
                        help="模型大小 (默认: large-v3)")
    parser.add_argument("--lang", "-l", default="zh",
                        help="语言代码，如 zh/en/ja，填 auto 自动检测 (默认: zh)")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                        help="批量大小，T4 GPU 建议 16 (默认: 16)")
    parser.add_argument("--no-vad", action="store_true",
                        help="关闭 VAD 静音过滤")
    parser.add_argument("--no-word-ts", action="store_true",
                        help="关闭词级时间戳")
    parser.add_argument("--no-srt", action="store_true",
                        help="不生成 SRT 字幕文件")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="输出目录（默认与音频同目录）")

    args = parser.parse_args()

    lang = None if args.lang == "auto" else args.lang

    transcribe(
        audio_path=args.audio,
        model_size=args.model,
        language=lang,
        batch_size=args.batch_size,
        vad_filter=not args.no_vad,
        word_timestamps=not args.no_word_ts,
        output_dir=args.out_dir,
        write_srt=not args.no_srt,
    )


if __name__ == "__main__":
    main()
