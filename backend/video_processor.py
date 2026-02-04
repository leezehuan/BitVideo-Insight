import os
import logging
import re
import shlex
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import yt_dlp

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器，使用yt-dlp下载和转换视频"""
    
    def __init__(self):
        self.ydl_opts = {
            'format': 'bestaudio/best',  # 优先下载最佳音频源
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # 直接在提取阶段转换为单声道 16k（空间小且稳定）
                'preferredcodec': 'm4a',
                'preferredquality': '192'
            }],
            # 全局FFmpeg参数：单声道 + 16k 采样率 + faststart
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,  # 强制只下载单个视频，不下载播放列表
        }
    
    def _is_youtube_url(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.netloc or "").lower()
            return "youtube.com" in host or "youtu.be" in host
        except Exception:
            return False

    def _normalize_lang_preferences(self, target_language: Optional[str]) -> list[str]:
        if not target_language:
            return []
        lang = target_language.lower().strip()
        prefs = [lang]
        if lang == "zh":
            prefs.extend(["zh-cn", "zh-hans", "zh-hant", "zh-tw", "zh-hk", "zh-hans-cn"]) 
        elif lang == "en":
            prefs.extend(["en-us", "en-gb", "en-ca", "en-au"]) 
        return list(dict.fromkeys([p for p in prefs if p]))

    def _pick_subtitle_format(self, formats: list[dict]) -> Optional[dict]:
        if not formats:
            return None
        preferred_exts = ["vtt", "srt", "ttml", "srv3", "srv2", "srv1"]
        for ext in preferred_exts:
            for f in formats:
                if (f.get("ext") or "").lower() == ext and f.get("url"):
                    return f
        for f in formats:
            if f.get("url"):
                return f
        return None

    def _format_time(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _parse_timecode_to_seconds(self, timecode: str) -> float:
        tc = (timecode or "").strip()
        tc = tc.replace(",", ".")
        parts = tc.split(":")
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = "0"
            m, s = parts
        else:
            return 0.0
        try:
            return float(h) * 3600 + float(m) * 60 + float(s)
        except Exception:
            return 0.0

    def _parse_vtt(self, content: str) -> list[tuple[float, float, str]]:
        lines = (content or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: list[tuple[float, float, str]] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip("\ufeff").strip()
            if not line or line.upper() == "WEBVTT":
                i += 1
                continue

            m = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{1,3})\s+-->\s+(\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{1,3})", line)
            if not m:
                i += 1
                continue

            start = self._parse_timecode_to_seconds(m.group(1))
            end = self._parse_timecode_to_seconds(m.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != "":
                t = lines[i].strip()
                if t and not t.startswith("NOTE"):
                    text_lines.append(t)
                i += 1
            text = " ".join(text_lines).strip()
            if text:
                segments.append((start, end, text))
            i += 1
        return segments

    def _parse_srt(self, content: str) -> list[tuple[float, float, str]]:
        text = (content or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        blocks = re.split(r"\n\s*\n", text)
        segments: list[tuple[float, float, str]] = []
        for block in blocks:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if len(lines) < 2:
                continue
            ts_line = lines[1] if re.match(r"^\d+$", lines[0]) else lines[0]
            m = re.match(r"^(\d{1,2}:\d{2}:\d{2}[\.,]\d{1,3})\s+-->\s+(\d{1,2}:\d{2}:\d{2}[\.,]\d{1,3})", ts_line)
            if not m:
                continue
            start = self._parse_timecode_to_seconds(m.group(1))
            end = self._parse_timecode_to_seconds(m.group(2))
            text_lines = lines[2:] if re.match(r"^\d+$", lines[0]) else lines[1:]
            seg_text = " ".join(text_lines).strip()
            if seg_text:
                segments.append((start, end, seg_text))
        return segments

    def _parse_subtitle_content(self, content: str, ext: str) -> tuple[list[tuple[float, float, str]], str]:
        e = (ext or "").lower()
        if e == "vtt":
            return self._parse_vtt(content), "vtt"
        if e == "srt":
            return self._parse_srt(content), "srt"
        return [], e

    def _normalize_lang_code(self, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        code = lang.lower().strip()
        if code.startswith("zh"):
            return "zh"
        if code.startswith("en"):
            return "en"
        if "-" in code:
            return code.split("-", 1)[0]
        return code

    def _segments_to_markdown_transcript(self, segments: list[tuple[float, float, str]], language: Optional[str]) -> str:
        detected_language = (language or "").strip() or "unknown"
        transcript_lines = []
        transcript_lines.append("# Video Transcription")
        transcript_lines.append("")
        transcript_lines.append(f"**Detected Language:** {detected_language}")
        transcript_lines.append("**Language Probability:** 1.00")
        transcript_lines.append("")
        transcript_lines.append("## Transcription Content")
        transcript_lines.append("")

        for start, end, text in segments:
            start_time = self._format_time(start)
            end_time = self._format_time(end)
            transcript_lines.append(f"**[{start_time} - {end_time}]**")
            transcript_lines.append("")
            transcript_lines.append(text.strip())
            transcript_lines.append("")

        return "\n".join(transcript_lines).strip() + "\n"

    async def try_get_youtube_subtitle_transcript(self, url: str, preferred_language: Optional[str] = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if not self._is_youtube_url(url):
            return None, None, None

        import asyncio
        def _extract_info():
            with yt_dlp.YoutubeDL({
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }) as ydl:
                return ydl.extract_info(url, download=False)

        info = await asyncio.to_thread(_extract_info)
        if not info:
            return None, None, None

        video_title = info.get("title", "unknown")
        subtitles = info.get("subtitles") or {}
        automatic_captions = info.get("automatic_captions") or {}

        preferred_langs = self._normalize_lang_preferences(preferred_language)
        all_langs_manual = list(subtitles.keys())
        all_langs_auto = list(automatic_captions.keys())

        selected_lang = None
        selected_formats = None
        for lang in preferred_langs:
            if lang in subtitles:
                selected_lang = lang
                selected_formats = subtitles.get(lang)
                break
        if not selected_lang and all_langs_manual:
            selected_lang = all_langs_manual[0]
            selected_formats = subtitles.get(selected_lang)

        if not selected_lang:
            for lang in preferred_langs:
                if lang in automatic_captions:
                    selected_lang = lang
                    selected_formats = automatic_captions.get(lang)
                    break
        if not selected_lang and all_langs_auto:
            selected_lang = all_langs_auto[0]
            selected_formats = automatic_captions.get(selected_lang)

        if not selected_lang or not selected_formats:
            return None, video_title, None

        normalized_lang = self._normalize_lang_code(selected_lang)

        fmt = self._pick_subtitle_format(selected_formats)
        if not fmt:
            return None, video_title, normalized_lang

        sub_url = fmt.get("url")
        ext = (fmt.get("ext") or "vtt").lower()
        if not sub_url:
            return None, video_title, normalized_lang

        def _download_subtitle_text() -> str:
            req = urllib.request.Request(
                sub_url,
                headers={"User-Agent": "Mozilla/5.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            try:
                return data.decode("utf-8")
            except Exception:
                return data.decode("utf-8", errors="ignore")

        content = await asyncio.to_thread(_download_subtitle_text)
        if not content:
            return None, video_title, normalized_lang

        segments, _ = self._parse_subtitle_content(content, ext)
        if not segments:
            cleaned = re.sub(r"\n{3,}", "\n\n", content.strip())
            transcript_lines = []
            transcript_lines.append("# Video Transcription")
            transcript_lines.append("")
            transcript_lines.append(f"**Detected Language:** {normalized_lang or 'unknown'}")
            transcript_lines.append("**Language Probability:** 1.00")
            transcript_lines.append("")
            transcript_lines.append("## Transcription Content")
            transcript_lines.append("")
            transcript_lines.append(cleaned)
            transcript_lines.append("")
            return "\n".join(transcript_lines).strip() + "\n", video_title, normalized_lang

        transcript_text = self._segments_to_markdown_transcript(segments, normalized_lang)
        return transcript_text, video_title, normalized_lang

    async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
        """
        下载视频并转换为m4a格式
        
        Args:
            url: 视频链接
            output_dir: 输出目录
            
        Returns:
            转换后的音频文件路径
        """
        try:
            # 创建输出目录
            output_dir.mkdir(exist_ok=True)
            
            # 生成唯一的文件名
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
            # 更新yt-dlp选项
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = output_template
            
            logger.info(f"开始下载视频: {url}")
            
            # 直接同步执行，不使用线程池
            # 在FastAPI中，IO密集型操作可以直接await
            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 获取视频信息（放到线程池避免阻塞事件循环）
                info = await asyncio.to_thread(ydl.extract_info, url, False)
                video_title = info.get('title', 'unknown')
                expected_duration = info.get('duration') or 0
                logger.info(f"视频标题: {video_title}")
                
                # 下载视频（放到线程池避免阻塞事件循环）
                await asyncio.to_thread(ydl.download, [url])
            
            # 查找生成的m4a文件
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
            if not os.path.exists(audio_file):
                # 如果m4a文件不存在，查找其他音频格式
                for ext in ['webm', 'mp4', 'mp3', 'wav']:
                    potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                else:
                    raise Exception("未找到下载的音频文件")
            
            # 校验时长，如果和源视频差异较大，尝试一次ffmpeg规范化重封装
            try:
                import subprocess, shlex
                probe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
                out = subprocess.check_output(probe_cmd, shell=True).decode().strip()
                actual_duration = float(out) if out else 0.0
            except Exception as _:
                actual_duration = 0.0
            
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"音频时长异常，期望{expected_duration}s，实际{actual_duration}s，尝试重封装修复…"
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    fix_cmd = f"ffmpeg -y -i {shlex.quote(audio_file)} -vn -c:a aac -b:a 160k -movflags +faststart {shlex.quote(fixed_path)}"
                    subprocess.check_call(fix_cmd, shell=True)
                    # 用修复后的文件替换
                    audio_file = fixed_path
                    # 重新探测
                    out2 = subprocess.check_output(probe_cmd.replace(shlex.quote(audio_file.rsplit('.',1)[0]+'.m4a'), shlex.quote(audio_file)), shell=True).decode().strip()
                    actual_duration2 = float(out2) if out2 else 0.0
                    logger.info(f"重封装完成，新时长≈{actual_duration2:.2f}s")
                except Exception as e:
                    logger.error(f"重封装失败：{e}")
            
            logger.info(f"音频文件已保存: {audio_file}")
            return audio_file, video_title
            
        except Exception as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise Exception(f"下载视频失败: {str(e)}")

    async def download_video(self, url: str, output_dir: Path) -> tuple[str, str]:
        """
        下载完整视频文件（用于截图）
        """
        try:
            output_dir.mkdir(exist_ok=True)

            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"video_{unique_id}.%(ext)s")
            ydl_opts = {
                "format": "bestvideo+bestaudio/best",
                "outtmpl": output_template,
                "merge_output_format": "mp4",
                "prefer_ffmpeg": True,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
            }

            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, url, False)
                video_title = info.get("title", "unknown") if info else "unknown"
                await asyncio.to_thread(ydl.download, [url])

            candidates = list(output_dir.glob(f"video_{unique_id}.*"))
            if not candidates:
                raise Exception("未找到下载的视频文件")
            video_path = str(sorted(candidates, key=lambda p: p.suffix != ".mp4")[0])
            logger.info(f"视频文件已保存: {video_path}")
            return video_path, video_title
        except Exception as e:
            logger.error(f"下载完整视频失败: {str(e)}")
            raise Exception(f"下载完整视频失败: {str(e)}")

    def _safe_timestamp_for_filename(self, timestamp: str) -> str:
        safe = re.sub(r"[^0-9:]", "", timestamp or "")
        return safe.replace(":", "-") or "unknown"

    def extract_screenshots(self, video_path: Path, screenshots: list[dict], output_dir: Path) -> list[dict]:
        """
        根据时间戳截图
        """
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return []
        output_dir.mkdir(exist_ok=True)

        results: list[dict] = []
        for idx, shot in enumerate(screenshots, start=1):
            timestamp = str(shot.get("timestamp", "")).strip()
            if not timestamp:
                continue
            safe_ts = self._safe_timestamp_for_filename(timestamp)
            filename = f"screenshot_{idx:02d}_{safe_ts}.jpg"
            output_path = output_dir / filename
            cmd = (
                f"ffmpeg -y -ss {shlex.quote(timestamp)} -i {shlex.quote(str(video_path))} "
                f"-frames:v 1 -q:v 2 {shlex.quote(str(output_path))}"
            )
            try:
                subprocess.check_call(cmd, shell=True)
                results.append({
                    "index": idx,
                    "timestamp": timestamp,
                    "reason": shot.get("reason", ""),
                    "filename": filename,
                    "path": str(output_path),
                    "label": f"图{idx}",
                })
            except Exception as e:
                logger.warning(f"截图失败 {timestamp}: {e}")
        return results
    
    def get_video_info(self, url: str) -> dict:
        """
        获取视频信息
        
        Args:
            url: 视频链接
            
        Returns:
            视频信息字典
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            raise Exception(f"获取视频信息失败: {str(e)}")
