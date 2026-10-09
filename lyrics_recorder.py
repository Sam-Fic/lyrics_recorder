#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""歌词分段录音工具

按换行拆分整首歌的歌词，每两行为一个录音单位；
选中单位后点击录音，产出同名的 .wav 音频与 .txt 歌词文本。

用法:
    python lyrics_recorder.py           启动图形界面
    python lyrics_recorder.py --web     启动 WebUI（局域网手机麦克风录音）
    python lyrics_recorder.py --selftest  无 GUI 自检（验证解析与文件生成）

依赖:
    sounddevice  numpy  （桌面录音时按需加载；tkinter 为系统自带）
    桌面端默认参数: 48000 Hz, 24 bit, 单声道
    WebUI 仅用标准库，无需额外依赖
"""

import argparse
import datetime
import ipaddress
import json
import re
import socket
import ssl
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "recordings"

SAMPLE_RATES = [44100, 48000, 32000, 24000, 22050]
BIT_DEPTHS = [16, 24, 32]
CHANNEL_OPTIONS = [1, 2]
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BIT_DEPTH = 24
DEFAULT_CHANNELS = 1


def parse_lyric_units(text: str) -> list[list[str]]:
    """按换行拆句，过滤空行，每两行组成一个录音单位。

    奇数行时最后一行单独成单位。
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [lines[i:i + 2] for i in range(0, len(lines), 2)]


def float32_to_pcm(data, bit_depth: int) -> bytes:
    """把 [-1, 1] 的 float32 数据编码为指定位深的 PCM 小端字节。"""
    import numpy as np

    data = np.clip(data, -1.0, 1.0)
    if bit_depth == 16:
        return (data * 32767.0).astype(np.int16).tobytes()
    if bit_depth == 24:
        arr = (data * 8388607.0).astype("<i4").view(np.uint8).reshape(-1, 4)
        return arr[:, :3].tobytes()
    if bit_depth == 32:
        return (data * 2147483647.0).astype(np.int32).tobytes()
    raise ValueError(f"不支持的位深: {bit_depth}")


def write_wav(path, data, samplerate: int, bit_depth: int, channels: int) -> None:
    """把 float32 录音数据写入 wav 文件。"""
    pcm = float32_to_pcm(data, bit_depth)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bit_depth // 8)
        wf.setframerate(samplerate)
        wf.writeframes(pcm)


_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f/\\:*?"<>|]')


def unit_filename(index: int, prefix: str = "") -> str:
    """生成录音单位的基文件名：有前缀时为 `prefix_001`，否则为 `001`。

    前缀会去掉文件系统非法字符（路径分隔符、控制字符、保留字符），并保留 Unicode。
    """
    prefix = (prefix or "").strip().strip(".")
    prefix = _UNSAFE_FILENAME_CHARS.sub("_", prefix).strip(".") if prefix else ""
    base = f"{index:03d}"
    return f"{prefix}_{base}" if prefix else base


def save_unit(output_dir, index: int, unit: list[str],
              data, samplerate: int, bit_depth: int, channels: int,
              prefix: str = "") -> tuple[Path, Path]:
    """保存一个单位的 wav（数据）与 txt（歌词原文，UTF-8）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = unit_filename(index, prefix)
    wav_path = output_dir / f"{name}.wav"
    txt_path = output_dir / f"{name}.txt"
    write_wav(wav_path, data, samplerate, bit_depth, channels)
    txt_path.write_text("\n".join(unit) + "\n", encoding="utf-8")
    return wav_path, txt_path


def save_unit_from_bytes(output_dir, index: int, unit: list[str], wav_bytes: bytes,
                         prefix: str = "") -> tuple[Path, Path]:
    """保存一个单位的 wav（前端上传的原始 WAV 字节）与 txt（歌词原文，UTF-8）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = unit_filename(index, prefix)
    wav_path = output_dir / f"{name}.wav"
    txt_path = output_dir / f"{name}.txt"
    wav_path.write_bytes(wav_bytes)
    txt_path.write_text("\n".join(unit) + "\n", encoding="utf-8")
    return wav_path, txt_path


class Recorder:
    """基于 sounddevice 的流式录音器。"""

    def __init__(self, samplerate: int, channels: int, device=None):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self._stream = None
        self._frames = []
        self.started_at = 0.0

    def start(self) -> None:
        import sounddevice as sd

        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()
        self.started_at = time.time()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            print("录音状态提示:", status, file=sys.stderr)
        self._frames.append(indata.copy())

    def stop(self):
        """停止并返回拼接后的 float32 数据；无数据时返回 None。"""
        import numpy as np

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return None
        return np.concatenate(self._frames, axis=0)

    @property
    def duration(self) -> float:
        return time.time() - self.started_at if self._stream is not None else 0.0


class RecorderApp:
    """tkinter 图形界面。"""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = root
        root.title("歌词分段录音工具（DiffSinger 声库采集）")
        root.geometry("760x680")

        self.units = []          # list[list[str]]
        self.current_index = None
        self.recorder = None
        self._timer_id = None

        # ---- 歌词输入区 ----
        lyric_frame = ttk.LabelFrame(root, text="歌词（一整首歌，按换行分句，每两行为一个录音单位）")
        lyric_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        btn_row = ttk.Frame(lyric_frame)
        btn_row.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(btn_row, text="从文件加载", command=self._load_lyrics_file).pack(side="left")
        ttk.Button(btn_row, text="解析歌词", command=self._parse_lyrics).pack(side="left", padx=6)
        ttk.Label(btn_row, text="（选择单位后点下方按钮录音）").pack(side="left", padx=10)

        text_wrap = ttk.Frame(lyric_frame)
        text_wrap.pack(fill="both", expand=True, padx=6, pady=6)
        self.lyric_text = tk.Text(text_wrap, height=8, wrap="char")
        scroll_y = ttk.Scrollbar(text_wrap, orient="vertical", command=self.lyric_text.yview)
        self.lyric_text.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.lyric_text.pack(side="left", fill="both", expand=True)

        # ---- 单位列表区 ----
        unit_frame = ttk.LabelFrame(root, text="录音单位")
        unit_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self.unit_list = tk.Listbox(unit_frame, height=8)
        unit_scroll = ttk.Scrollbar(unit_frame, orient="vertical", command=self.unit_list.yview)
        self.unit_list.configure(yscrollcommand=unit_scroll.set)
        unit_scroll.pack(side="right", fill="y")
        self.unit_list.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.unit_list.bind("<<ListboxSelect>>", self._on_select_unit)

        # ---- 参数区 ----
        opt_frame = ttk.LabelFrame(root, text="录音参数（默认: 48kHz / 24bit / 单声道）")
        opt_frame.pack(fill="x", padx=8, pady=4)

        self.var_rate = tk.StringVar(value=str(DEFAULT_SAMPLE_RATE))
        self.var_bits = tk.StringVar(value=str(DEFAULT_BIT_DEPTH))
        self.var_channels = tk.StringVar(value=str(DEFAULT_CHANNELS))
        self.var_outdir = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.var_device = tk.StringVar(value="默认（系统）")
        self.var_prefix = tk.StringVar(value="")
        self.devices = {}  # 显示名 -> device index（None 表示系统默认）

        ttk.Label(opt_frame, text="采样率 (Hz):").grid(row=0, column=0, sticky="e", padx=(10, 4), pady=4)
        ttk.Combobox(opt_frame, textvariable=self.var_rate, values=[str(v) for v in SAMPLE_RATES],
                     width=8, state="readonly").grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(opt_frame, text="位深 (bit):").grid(row=0, column=2, sticky="e", padx=(16, 4), pady=4)
        ttk.Combobox(opt_frame, textvariable=self.var_bits, values=[str(v) for v in BIT_DEPTHS],
                     width=8, state="readonly").grid(row=0, column=3, sticky="w", pady=4)
        ttk.Label(opt_frame, text="声道:").grid(row=0, column=4, sticky="e", padx=(16, 4), pady=4)
        ttk.Combobox(opt_frame, textvariable=self.var_channels, values=[str(v) for v in CHANNEL_OPTIONS],
                     width=8, state="readonly").grid(row=0, column=5, sticky="w", pady=4)

        ttk.Label(opt_frame, text="输入设备:").grid(row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        self.device_combo = ttk.Combobox(opt_frame, textvariable=self.var_device,
                                         values=["默认（系统）"], width=40, state="readonly")
        self.device_combo.grid(row=1, column=1, columnspan=4, sticky="we", pady=4)
        ttk.Button(opt_frame, text="刷新设备", command=self._load_input_devices).grid(
            row=1, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(opt_frame, text="输出目录:").grid(row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        ttk.Entry(opt_frame, textvariable=self.var_outdir, width=52).grid(row=2, column=1, columnspan=4,
                                                                          sticky="we", pady=4)
        ttk.Button(opt_frame, text="浏览…", command=self._choose_output_dir).grid(
            row=2, column=5, sticky="w", padx=4, pady=4)

        ttk.Label(opt_frame, text="文件名前缀:").grid(row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        ttk.Entry(opt_frame, textvariable=self.var_prefix, width=24).grid(row=3, column=1, columnspan=3,
                                                                          sticky="we", pady=4)
        ttk.Label(opt_frame, text="（留空则 001；填写则 prefix_001）").grid(row=3, column=4, columnspan=2,
                                                                          sticky="w", padx=(6, 0), pady=4)
        opt_frame.columnconfigure(1, weight=1)

        self._load_input_devices()

        # ---- 录音控制区 ----
        rec_frame = ttk.Frame(root)
        rec_frame.pack(fill="x", padx=8, pady=6)

        self.cur_label = ttk.Label(rec_frame, text="当前单位: （请先在列表中选一个单位）")
        self.cur_label.pack(side="left", padx=4)
        self.record_btn = tk.Button(rec_frame, text="开始录音", width=12,
                                    command=self._toggle_record, bg="#4caf50", fg="white",
                                    font=("", 11, "bold"))
        self.record_btn.pack(side="right", padx=4)
        self.duration_label = ttk.Label(rec_frame, text="00:00.0")
        self.duration_label.pack(side="right", padx=12)

        # ---- 状态栏 ----
        self.status_label = ttk.Label(root, text="就绪", relief="sunken", anchor="w")
        self.status_label.pack(fill="x", padx=8, pady=(0, 8))

        self._set_status("粘贴或加载歌词文件，点击「解析歌词」，再选中单位开始录音。")

    # ---------- 交互逻辑 ----------

    def _load_lyrics_file(self) -> None:
        path = self.filedialog.askopenfilename(
            title="选择歌词文本文件", filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            content = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            self.messagebox.showerror("读取失败", f"无法读取 {path}\n{exc}")
            return
        self.lyric_text.delete("1.0", "end")
        self.lyric_text.insert("1.0", content)
        self._set_status(f"已加载 {path}，点击「解析歌词」。")

    def _parse_lyrics(self) -> None:
        if self.recorder is not None:
            self.messagebox.showwarning("录音中", "请先停止录音再解析歌词。")
            return
        text = self.lyric_text.get("1.0", "end")
        self.units = parse_lyric_units(text)
        self.unit_list.delete(0, "end")
        for idx, unit in enumerate(self.units, start=1):
            preview = " / ".join(unit)
            self.unit_list.insert("end", f"{idx:03d}  {preview[:60]}")
        self.current_index = None
        self.cur_label.config(text="当前单位: （请先在列表中选一个单位）")
        if not self.units:
            self._set_status("未解析出有效歌词行（空行会被忽略）。")
        else:
            self._set_status(f"解析出 {len(self.units)} 个录音单位。")

    def _on_select_unit(self, _event=None) -> None:
        if self.recorder is not None:
            return
        selection = self.unit_list.curselection()
        if not selection:
            return
        self.current_index = selection[0]
        unit = self.units[self.current_index]
        self.cur_label.config(text=f"当前单位: {self.current_index + 1:03d}  «{unit[0]}»")

    def _choose_output_dir(self) -> None:
        path = self.filedialog.askdirectory(title="选择输出目录")
        if path:
            self.var_outdir.set(path)

    def _load_input_devices(self) -> None:
        """枚举系统的输入设备并填充下拉框；无 PortAudio 时降级为仅默认项。"""
        self.devices = {}
        values = ["默认（系统）"]
        try:
            import sounddevice as sd

            for idx, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    label = f"{idx}: {dev['name']}"
                    self.devices[label] = idx
                    values.append(label)
        except Exception as exc:
            self._set_status(f"无法枚举音频设备（{exc}）；将使用系统默认输入。")
        self.device_combo.configure(values=values)
        if self.var_device.get() not in values:
            self.var_device.set("默认（系统）")

    def _toggle_record(self) -> None:
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_and_save()

    def _start_recording(self) -> None:
        if self.current_index is None:
            self.messagebox.showwarning("未选单位", "请先在列表中选择一个录音单位。")
            return
        try:
            samplerate = int(self.var_rate.get())
            channels = int(self.var_channels.get())
        except ValueError:
            self.messagebox.showerror("参数错误", "采样率或声道数不合法。")
            return
        device = self.devices.get(self.var_device.get(), None)
        try:
            self.recorder = Recorder(samplerate, channels, device=device)
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            self.messagebox.showerror("录音启动失败",
                                      f"{exc}\n\n请检查麦克风设备与 sounddevice 是否安装正确。")
            return
        self.record_btn.config(text="停止并保存", bg="#f44336")
        self.cur_label.config(text=f"录音中: {' / '.join(self.units[self.current_index])}")
        self._set_status("录音中… 再次点击按钮停止并保存。")
        self._tick_duration()

    def _tick_duration(self) -> None:
        if self.recorder is None:
            return
        seconds = self.recorder.duration
        self.duration_label.config(text=f"{int(seconds // 60):02d}:{seconds % 60:04.1f}")
        self._timer_id = self.root.after(100, self._tick_duration)

    def _stop_and_save(self) -> None:
        data = self.recorder.stop()
        self._timer_id and self.root.after_cancel(self._timer_id)
        self.recorder = None
        self.duration_label.config(text="00:00.0")
        self.record_btn.config(text="开始录音", bg="#4caf50")
        if data is None or len(data) == 0:
            self._set_status("未录到有效音频，未保存。")
            return
        try:
            samplerate = int(self.var_rate.get())
            bit_depth = int(self.var_bits.get())
            channels = int(self.var_channels.get())
            wav_path, txt_path = save_unit(
                self.var_outdir.get(), self.current_index + 1,
                self.units[self.current_index], data, samplerate, bit_depth, channels,
                prefix=self.var_prefix.get())
        except Exception as exc:
            self.messagebox.showerror("保存失败", str(exc))
            self._set_status("保存失败。")
            return
        self._set_status(f"已保存 {wav_path}\n{wav_path.name} 与 {txt_path.name}（{len(data) / samplerate:.2f}s）")
        self.messagebox.showinfo("已保存", f"{wav_path.name} 与 {txt_path.name}\n{wav_path.parent}")

    def _set_status(self, text: str) -> None:
        self.status_label.config(text=text)


# ---------- 自检 ----------

def run_selftest() -> int:
    """无 GUI、无声卡的链路自检：解析逻辑 + wav/txt 生成与回读校验。"""
    import tempfile

    import numpy as np

    text = "第一行歌词\n第二行歌词\n\n第三行歌词\n第四行歌词\n以后还有第五行\n"
    units = parse_lyric_units(text)
    assert len(units) == 3, f"应拆成 3 个单位，实际 {len(units)}: {units}"
    assert units[0] == ["第一行歌词", "第二行歌词"]
    assert units[2] == ["以后还有第五行"], "奇数行应单独成单位"

    out_dir = Path(tempfile.mkdtemp(prefix="lyrics_recorder_selftest_"))
    sr, bits, channels = DEFAULT_SAMPLE_RATE, DEFAULT_BIT_DEPTH, DEFAULT_CHANNELS
    t = np.arange(int(sr * 1.0)) / sr
    data = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    for idx, unit in enumerate(units, start=1):
        save_unit(out_dir, idx, unit, data, sr, bits, channels)

    for idx in (1, 2, 3):
        wav_path = out_dir / f"{idx:03d}.wav"
        txt_path = out_dir / f"{idx:03d}.txt"
        assert wav_path.exists() and txt_path.exists(), f"缺少 {idx:03d} 的文件"
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == channels
            assert wf.getframerate() == sr
            assert wf.getsampwidth() == bits // 8
            assert wf.getnframes() == len(data), "wav 帧数应与录音一致"
        assert txt_path.read_text(encoding="utf-8") == "\n".join(units[idx - 1]) + "\n"

    # 24bit / 32bit 头部参数回读
    for b in (24, 32):
        probe = out_dir / f"probe_{b}.wav"
        write_wav(probe, data, sr, b, channels)
        with wave.open(str(probe), "rb") as wf:
            assert wf.getsampwidth() == b // 8
            assert wf.getframerate() == sr
            assert wf.getnframes() == len(data)

    # 文件名前缀
    save_unit(out_dir, 9, ["前缀测试A", "前缀测试B"], data, sr, bits, channels, prefix="demo")
    assert (out_dir / "demo_009.wav").exists() and (out_dir / "demo_009.txt").exists()
    # 非法字符（如路径分隔符）应被安全过滤
    save_unit(out_dir, 10, ["x", "y"], data, sr, bits, channels, prefix="a/b:c*")
    assert (out_dir / "a_b_c__010.wav").exists(), "前缀中的非法字符应被替换为下划线"

    print(f"selftest OK: 3 个单位的解析与 wav/txt 生成全部通过")
    print(f"产物样例目录: {out_dir}")
    return 0


WEB_MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB，防止异常超大上传


def _make_web_handler(output_dir):
    """构造 WebUI 请求处理器（闭包携带 output_dir 与页面内容）。"""
    html_path = BASE_DIR / "webui.html"
    if html_path.exists():
        html_content = html_path.read_text(encoding="utf-8")
    else:
        html_content = "<h1>webui.html 缺失：请确认与 lyrics_recorder.py 同目录</h1>"

    class WebHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self):
            # 每次读取最新文件，便于修改前端后无需重启服务
            if html_path.exists():
                body = html_path.read_text(encoding="utf-8").encode("utf-8")
            else:
                body = "<h1>webui.html 缺失：请确认与 lyrics_recorder.py 同目录</h1>".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_asset(self, path, ctype):
            body = path.read_bytes() if path.exists() else b""
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send_html()
            elif parsed.path == "/api/config":
                self._send_json({
                    "sample_rates": SAMPLE_RATES,
                    "bit_depths": BIT_DEPTHS,
                    "channels": CHANNEL_OPTIONS,
                    "default_sample_rate": DEFAULT_SAMPLE_RATE,
                    "default_bit_depth": DEFAULT_BIT_DEPTH,
                    "default_channels": DEFAULT_CHANNELS,
                    "output_dir": str(Path(output_dir)),
                })
            elif parsed.path == "/m3e.min.js":
                self._send_asset(BASE_DIR / "m3e.min.js", "text/javascript; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/parse":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length else b""
                    payload = json.loads(raw.decode("utf-8"))
                    units = parse_lyric_units(payload.get("text", ""))
                    self._send_json({"units": units})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
                return
            if parsed.path == "/api/upload":
                self._handle_upload(parsed)
                return
            self._send_json({"error": "not found"}, 404)

        def _handle_upload(self, parsed):
            qs = parse_qs(parsed.query)
            try:
                index = int(qs.get("index", ["0"])[0])
                rate = int(qs.get("rate", ["0"])[0])
                bits = int(qs.get("bits", ["16"])[0])
                channels = int(qs.get("channels", ["1"])[0])
                text = qs.get("text", [""])[0]
                prefix = qs.get("prefix", [""])[0]
            except (ValueError, IndexError) as exc:
                self._send_json({"ok": False, "error": f"参数错误: {exc}"}, 400)
                return
            if index <= 0:
                self._send_json({"ok": False, "error": "index 不合法"}, 400)
                return
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > WEB_MAX_CONTENT_LENGTH:
                self._send_json({"ok": False, "error": "音频数据为空或过大"}, 400)
                return
            wav_bytes = self.rfile.read(length)
            unit = text.split("\n")
            try:
                wav_path, txt_path = save_unit_from_bytes(
                    output_dir, index, unit, wav_bytes, prefix=prefix)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
                return
            duration = 0.0
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    duration = wf.getnframes() / wf.getframerate()
            except Exception:
                pass
            self._send_json({
                "ok": True,
                "wav": wav_path.name,
                "txt": txt_path.name,
                "duration": duration,
            })

        def log_message(self, fmt, *args):
            sys.stderr.write("[web] " + (fmt % args) + "\n")

    return WebHandler


def get_lan_ip() -> str:
    """尽力获取局域网 IP（用于打印手机访问地址）。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        sock.close()
    return ip


def generate_self_signed_cert(cert_path: Path, key_path: Path, host: str) -> None:
    """生成自签名证书（含 localhost 与 host 的 SAN），用于 HTTPS 安全上下文。

    浏览器对麦克风/摄像头要求在安全上下文（HTTPS 或 localhost）下访问，
    局域网内以 IP 访问时必须用 HTTPS，否则 navigator.mediaDevices 为 undefined。
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    san = [x509.DNSName("localhost")]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(host)))
    except ValueError:
        san.append(x509.DNSName(host))

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "lyrics-recorder")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM,
                         serialization.PrivateFormat.TraditionalOpenSSL,
                         serialization.NoEncryption())
    )


def run_web(host: str, port: int, output_dir, use_https: bool = False,
            cert: str = None, key: str = None) -> int:
    """启动 WebUI 服务（局域网手机麦克风录音）。

    手机以局域网 IP 访问时必须是 HTTPS，否则 getUserMedia 不可用；
    桌面以 localhost 访问时 HTTP 即可（localhost 视为安全上下文）。
    """
    handler = _make_web_handler(output_dir)
    server = ThreadingHTTPServer((host, port), handler)
    lan = get_lan_ip()
    scheme = "http"
    if use_https:
        scheme = "https"
        cert_path = Path(cert) if cert else (BASE_DIR / "webui_cert.pem")
        key_path = Path(key) if key else (BASE_DIR / "webui_key.pem")
        if not (cert_path.exists() and key_path.exists()):
            print("正在生成自签名证书（首次启动）…")
            generate_self_signed_cert(cert_path, key_path, lan)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
    print("歌词录音 WebUI 已启动：")
    print(f"  本机访问:  {scheme}://localhost:{port}")
    print(f"  手机访问:  {scheme}://{lan}:{port}   （手机需与本机同一局域网）")
    if use_https:
        print("  注意：自签名证书会触发浏览器安全警告，请点「高级 → 继续访问」。")
    print(f"  输出目录:  {Path(output_dir)}")
    print("  按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="歌词分段录音工具（DiffSinger 声库采集）")
    parser.add_argument("--selftest", action="store_true", help="无 GUI 自检（解析与文件生成链路）")
    parser.add_argument("--web", action="store_true", help="启动 WebUI（局域网手机麦克风录音）")
    parser.add_argument("--host", default="0.0.0.0", help="WebUI 绑定地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="WebUI 端口（默认 8000）")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTPUT_DIR), help="输出目录（默认 ./recordings）")
    parser.add_argument("--https", action="store_true",
                        help="启用 HTTPS（手机以局域网 IP 访问麦克风时必须开启）")
    parser.add_argument("--cert", default=None, help="HTTPS 证书路径（默认自动生成自签名证书）")
    parser.add_argument("--key", default=None, help="HTTPS 私钥路径（默认自动生成）")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftest()
    if args.web:
        return run_web(args.host, args.port, args.outdir,
                       use_https=args.https, cert=args.cert, key=args.key)

    try:
        import tkinter as tk  # noqa: F401
    except ImportError:
        print("缺少 tkinter：请安装系统图形库（如 Ubuntu: sudo apt install python3-tk）",
              file=sys.stderr)
        return 2

    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())