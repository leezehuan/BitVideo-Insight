import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import openai

from config_loader import load_config

logger = logging.getLogger(__name__)

class Summarizer:
    """文本总结器，使用OpenAI API生成多语言摘要"""
    
    def __init__(self):
        """初始化总结器"""
        # 从环境变量获取OpenAI API配置（优化/摘要分离）
        default_api_key = os.getenv("OPENAI_API_KEY")
        default_base_url = os.getenv("OPENAI_BASE_URL")

        optimize_api_key = os.getenv("OPENAI_OPTIMIZE_API_KEY") or default_api_key
        optimize_base_url = os.getenv("OPENAI_OPTIMIZE_BASE_URL") or default_base_url
        summarize_api_key = os.getenv("OPENAI_SUMMARIZE_API_KEY") or default_api_key
        summarize_base_url = os.getenv("OPENAI_SUMMARIZE_BASE_URL") or default_base_url

        self.optimize_model = os.getenv("OPENAI_OPTIMIZE_MODEL", "gpt-3.5-turbo")
        self.summarize_model = os.getenv("OPENAI_SUMMARIZE_MODEL", "gpt-4o")

        self.optimize_client = self._init_openai_client(
            optimize_api_key,
            optimize_base_url,
            "转录优化",
            "OPENAI_OPTIMIZE_API_KEY/OPENAI_API_KEY",
        )
        self.summarize_client = self._init_openai_client(
            summarize_api_key,
            summarize_base_url,
            "摘要",
            "OPENAI_SUMMARIZE_API_KEY/OPENAI_API_KEY",
        )

        config = load_config()
        self.prompt_config = {}
        if isinstance(config, dict):
            prompts = config.get("prompts", {})
            if isinstance(prompts, dict):
                self.prompt_config = prompts.get("summarizer", {}) or {}
        
        # 支持的语言映射
        self.language_map = {
            "en": "English",
            "zh": "中文（简体）",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어",
            "ar": "العربية"
        }

    def _init_openai_client(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        label: str,
        env_hint: str,
    ) -> Optional[openai.OpenAI]:
        if not api_key:
            logger.warning(f"未设置{env_hint}环境变量，{label}功能不可用")
            return None
        if base_url:
            logger.info(f"{label} OpenAI客户端已初始化，使用自定义端点: {base_url}")
            return openai.OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"{label} OpenAI客户端已初始化，使用默认端点")
        return openai.OpenAI(api_key=api_key)

    def _get_prompt_value(self, *keys: str, default: str) -> str:
        data: Any = self.prompt_config
        for key in keys:
            if not isinstance(data, dict):
                return default
            data = data.get(key)
            if data is None:
                return default
        if isinstance(data, list):
            parts = [str(item) for item in data if str(item).strip()]
            return "\n".join(parts) if parts else default
        if isinstance(data, str) and data.strip():
            return data
        return default

    def _render_prompt(self, template: str, **kwargs: Any) -> str:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    
    async def optimize_transcript(self, raw_transcript: str) -> str:
        """
        优化转录文本：修正错别字，保持原有格式
        为每句话添加标准时间戳，不进行分段、分块或合并
        
        Args:
            raw_transcript: 原始转录文本
            
        Returns:
            优化后的转录文本（Markdown格式）
        """
        try:
            if not self.optimize_client:
                logger.warning("优化API不可用，返回原始转录")
                return raw_transcript

            # 预处理：仅移除元信息，保留时间戳以便生成句级时间标签
            preprocessed = self._remove_timestamps_and_meta(raw_transcript)
            detected_lang_code = self._detect_transcript_language(preprocessed)
            if len(preprocessed) > 4000:
                logger.info("文本较长(%s chars)，仍使用单次优化（不分段）", len(preprocessed))
            return await self._format_single_chunk(preprocessed, detected_lang_code)

        except Exception as e:
            logger.error(f"优化转录文本失败: {str(e)}")
            logger.info("返回原始转录文本")
            return raw_transcript

    async def _format_single_chunk(self, chunk_text: str, transcript_language: str = 'zh') -> str:
        """单块优化（修正+格式化），遵循4000 tokens 限制。"""
        # 构建与JS版一致的系统/用户提示
        if transcript_language == 'zh':
            default_user = (
                "请对以下音频转录文本进行智能优化，要求：\n\n"
                "**内容优化（正确性优先）：**\n"
                "1. 错误修正（转录错误/错别字/同音字/专有名词）\n"
                "2. 适度改善语法，补全不完整句子，保持原意和语言不变\n"
                "3. 口语处理：保留自然口语与重复表达，不要删减内容，仅添加必要标点\n"
                "4. **绝对不要改变人称代词（I/我、you/你等）和说话者视角**\n\n"
                "**时间戳规则：**\n"
                "- 依据原始时间范围，为每一句生成一个标准时间戳\n"
                "- 时间戳格式统一为 [HH:MM:SS]，使用该句对应片段的起始时间\n"
                "- 不保留原始 **[start - end]** 时间范围行\n\n"
                "**输出格式：**\n"
                "- 每句话单独一行，以 [HH:MM:SS] 开头，后跟空格和句子\n"
                "- 不新增段落或标题，仅输出正文\n\n"
                "原始转录文本：\n{chunk_text}"
            )
            default_system = (
                "你是专业的音频转录文本优化助手，修正错误、改善通顺度，"
                "必须保持原意，不得删减口语/重复/细节；仅移除时间戳或元信息。"
                "需要根据原始时间范围，为每句话生成标准时间戳并置于句首，格式为[HH:MM:SS]。"
                "不要新增段落或合并内容，保持原句顺序。"
                "绝对不要改变人称代词或说话者视角。这可能是访谈对话，访谈者用'you'，被访者用'I/we'。"
            )

            prompt = self._render_prompt(
                self._get_prompt_value("optimize_format", "zh", "user", default=default_user),
                chunk_text=chunk_text,
            )
            system_prompt = self._render_prompt(
                self._get_prompt_value("optimize_format", "zh", "system", default=default_system)
            )
        else:
            default_user = (
                "Please intelligently optimize the following audio transcript text:\n\n"
                "Content Optimization (Accuracy First):\n"
                "1. Error Correction (typos, homophones, proper nouns)\n"
                "2. Moderate grammar improvement, complete incomplete sentences, keep original language/meaning\n"
                "3. Speech processing: keep natural fillers and repetitions, do NOT remove content; only add punctuation if needed\n"
                "4. **NEVER change pronouns (I, you, he, she, etc.) or speaker perspective**\n\n"
                "Timestamp Rules:\n"
                "- For every sentence, add a standard timestamp derived from the original time range\n"
                "- Use the format [HH:MM:SS] and pick the start time of the sentence's source segment\n"
                "- Remove original **[start - end]** range lines\n\n"
                "Output Format:\n"
                "- One sentence per line, prefixed by [HH:MM:SS] and a space\n"
                "- Do not add headings or new paragraphs; output plain text only\n\n"
                "Original transcript text:\n{chunk_text}"
            )
            default_system = (
                "You are a professional transcript assistant. Fix errors and improve fluency "
                "without changing meaning or removing any content; only timestamps/meta may be removed. "
                "You must add a standard timestamp for every sentence at the beginning, formatted as [HH:MM:SS], "
                "derived from the original time ranges. Do not add new paragraphs or merge content; keep the sentence order. "
                "NEVER change pronouns or speaker perspective. This may be an interview: interviewer uses 'you', interviewee uses 'I/we'."
            )

            prompt = self._render_prompt(
                self._get_prompt_value("optimize_format", "en", "user", default=default_user),
                chunk_text=chunk_text,
            )
            system_prompt = self._render_prompt(
                self._get_prompt_value("optimize_format", "en", "system", default=default_system)
            )

        try:
            response = self.optimize_client.chat.completions.create(
                model=self.optimize_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # 对齐JS：优化/格式化阶段最大tokens≈4000
                temperature=0.1
            )
            optimized_text = response.choices[0].message.content or ""
            optimized_text = self._remove_transcript_heading(optimized_text)
            return optimized_text.strip()
        except Exception as e:
            logger.error(f"单块文本优化失败: {e}")
            return self._remove_transcript_heading(chunk_text).strip()

    def _remove_timestamps_and_meta(self, text: str) -> str:
        """仅移除明显元信息（标题、检测语言等），保留时间戳行供后续处理。"""
        lines = text.split('\n')
        kept = []
        for line in lines:
            s = line.strip()
            # 跳过时间戳与元信息
            if s.startswith('#'):
                # 跳过标题行（通常是视频标题或章节标题）
                continue
            if s.startswith('**检测语言:**') or s.startswith('**语言概率:**'):
                continue
            kept.append(line)
        # 规范空行
        cleaned = '\n'.join(kept)
        return cleaned

    def _remove_transcript_heading(self, text: str) -> str:
        """移除开头或段落中的以 Transcript 为标题的行（任意级别#），不改变正文。"""
        if not text:
            return text
        import re
        # 移除形如 '## Transcript'、'# Transcript Text'、'### transcript' 的标题行
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{1,6}\s*transcript(\s+text)?\s*$", stripped, flags=re.I):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def get_screenshot_count(self) -> int:
        value = self.prompt_config.get("screenshot_count") if isinstance(self.prompt_config, dict) else None
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = 0
        return max(0, count)

    def _normalize_timestamp(self, timestamp: str) -> Optional[str]:
        ts = (timestamp or "").strip().strip("[]")
        if not ts:
            return None
        parts = ts.split(":")
        if len(parts) == 2:
            mm, ss = parts
            return f"00:{int(mm):02d}:{int(ss):02d}"
        if len(parts) == 3:
            hh, mm, ss = parts
            return f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        return None

    def _extract_timestamped_lines(self, text: str) -> list[dict]:
        lines = []
        pattern = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)$")
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            normalized = self._normalize_timestamp(match.group(1))
            sentence = match.group(2).strip()
            if normalized and sentence:
                lines.append({"timestamp": normalized, "text": sentence})
        return lines

    def strip_sentence_timestamps(self, text: str) -> str:
        pattern = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)$")
        cleaned_lines = []
        for raw_line in (text or "").splitlines():
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if re.match(r"^\*\*\[.+\]\*\*$", line):
                continue
            match = pattern.match(line)
            if match:
                cleaned_lines.append(match.group(2).strip())
            else:
                cleaned_lines.append(raw_line)
        return "\n".join(cleaned_lines).strip()

    async def request_screenshot_plan(self, timestamped_text: str, screenshot_count: Optional[int] = None) -> list[dict]:
        count = screenshot_count if screenshot_count is not None else self.get_screenshot_count()
        if count <= 0:
            return []
        client = self.summarize_client or self.optimize_client
        if not client:
            logger.warning("截图规划API不可用，跳过截图请求")
            return []

        default_system = (
            "你是视频截图规划助手。请基于全文每句的时间戳，挑选最关键的 {screenshot_count} 个截图时间点。\n"
            "输出必须是严格 JSON 数组，不要包含任何额外文字。\n"
            "每个元素格式为：{\"timestamp\": \"HH:MM:SS\", \"reason\": \"简短原因\"}。\n"
            "timestamp 必须来自原文已有的时间戳。"
        )
        default_user = (
            "以下是按句时间戳的完整文本：\n\n{transcript}\n\n"
            "请返回 {screenshot_count} 个截图时间点的 JSON 数组，只输出 JSON。"
        )

        system_prompt = self._render_prompt(
            self._get_prompt_value("screenshot_request", "system", default=default_system),
            screenshot_count=count,
        )
        user_prompt = self._render_prompt(
            self._get_prompt_value("screenshot_request", "user", default=default_user),
            screenshot_count=count,
            transcript=timestamped_text,
        )

        model = self.summarize_model if client is self.summarize_client else self.optimize_model
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=800,
                temperature=0.2,
            )
            content = response.choices[0].message.content or ""
            return self._parse_screenshot_plan(content, timestamped_text, count)
        except Exception as exc:
            logger.error(f"截图规划失败: {exc}")
            return self._fallback_screenshot_plan(timestamped_text, count)

    def _parse_screenshot_plan(self, response_text: str, transcript: str, screenshot_count: int) -> list[dict]:
        parsed = None
        try:
            parsed = json.loads(response_text)
        except Exception:
            start = response_text.find("[")
            end = response_text.rfind("]")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(response_text[start : end + 1])
                except Exception:
                    parsed = None

        available = []
        seen = set()
        for item in self._extract_timestamped_lines(transcript):
            ts = item["timestamp"]
            if ts not in seen:
                seen.add(ts)
                available.append(ts)

        if not isinstance(parsed, list):
            return self._fallback_screenshot_plan(transcript, screenshot_count)

        results = []
        available_set = set(available)
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            ts = self._normalize_timestamp(str(entry.get("timestamp", "")))
            if not ts or ts not in available_set:
                continue
            results.append({
                "timestamp": ts,
                "reason": str(entry.get("reason", "")).strip(),
            })
            if len(results) >= screenshot_count:
                break

        if len(results) < screenshot_count:
            fallback = self._fallback_screenshot_plan(transcript, screenshot_count)
            existing = {item["timestamp"] for item in results}
            for entry in fallback:
                if entry["timestamp"] not in existing:
                    results.append(entry)
                    existing.add(entry["timestamp"])
                if len(results) >= screenshot_count:
                    break

        return results

    def _fallback_screenshot_plan(self, transcript: str, screenshot_count: int) -> list[dict]:
        lines = self._extract_timestamped_lines(transcript)
        if not lines:
            return []
        unique = []
        seen = set()
        for item in lines:
            ts = item["timestamp"]
            if ts not in seen:
                unique.append(ts)
                seen.add(ts)
        if not unique:
            return []
        count = min(screenshot_count, len(unique))
        step = max(1, len(unique) / count)
        results = []
        for i in range(count):
            idx = int(i * step)
            ts = unique[min(idx, len(unique) - 1)]
            results.append({"timestamp": ts, "reason": "auto"})
        return results

    def annotate_transcript_with_screenshots(
        self,
        timestamped_text: str,
        screenshots: list[dict],
        image_url_prefix: Optional[str] = None,
    ) -> str:
        mapping: dict[str, list[dict]] = {}
        for shot in screenshots:
            ts = self._normalize_timestamp(str(shot.get("timestamp", "")))
            if not ts:
                continue
            mapping.setdefault(ts, []).append(shot)

        pattern = re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)$")
        output_lines = []
        for raw_line in (timestamped_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                output_lines.append("")
                continue
            if re.match(r"^\*\*\[.+\]\*\*$", line):
                continue
            match = pattern.match(line)
            if not match:
                output_lines.append(raw_line)
                continue
            ts = self._normalize_timestamp(match.group(1))
            sentence = match.group(2).strip()
            shots = mapping.get(ts, []) if ts else []
            if shots:
                markers = " ".join([f"(图{shot.get('index')})" for shot in shots if shot.get("index")])
                if markers:
                    sentence = f"{sentence} {markers}".strip()
            output_lines.append(sentence)
            if image_url_prefix:
                for shot in shots:
                    filename = shot.get("filename")
                    if not filename:
                        continue
                    label = shot.get("label") or f"图{shot.get('index', '')}".strip()
                    image_url = f"{image_url_prefix}/{filename}"
                    output_lines.append(f"![{label}]({image_url})")
            if shots:
                output_lines.append("")
        return "\n".join(output_lines).strip()

    def build_image_payloads(self, screenshots: list[dict]) -> list[dict]:
        payloads = []
        for shot in screenshots:
            path_value = shot.get("path")
            if not path_value:
                continue
            data_uri = self._encode_image(Path(path_value))
            if not data_uri:
                continue
            payloads.append({
                "data_uri": data_uri,
                "timestamp": shot.get("timestamp"),
                "label": shot.get("label"),
            })
        return payloads

    def _encode_image(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
            encoded = base64.b64encode(data).decode("ascii")
            ext = path.suffix.lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return f"data:{mime};base64,{encoded}"
        except Exception as exc:
            logger.warning(f"图片编码失败: {exc}")
            return None

    async def summarize(
        self,
        transcript: str,
        target_language: str = "zh",
        video_title: str = None,
        image_payloads: Optional[list[dict]] = None,
    ) -> str:
        """
        生成视频转录的摘要
        
        Args:
            transcript: 转录文本
            target_language: 目标语言代码
            
        Returns:
            摘要文本（Markdown格式）
        """
        try:
            if not self.summarize_client:
                logger.warning("摘要API不可用，生成备用摘要")
                return self._generate_fallback_summary(transcript, target_language, video_title)
            
            # 直接使用单次摘要（不分段/合并）
            return await self._summarize_single_text(
                transcript,
                target_language,
                video_title,
                image_payloads,
            )
            
        except Exception as e:
            logger.error(f"生成摘要失败: {str(e)}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_text(
        self,
        transcript: str,
        target_language: str,
        video_title: str = None,
        image_payloads: Optional[list[dict]] = None,
    ) -> str:
        """
        对单个文本进行摘要
        """
        # 获取目标语言名称
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # 构建英文提示词，适用于所有目标语言
        default_system = """You are a professional content analyst. Please generate a comprehensive summary in {language_name} for the following text.

Summary Requirements:
1. Extract the main topics and core viewpoints from the text
2. Maintain clear logical structure, highlighting the core arguments
3. Include important discussions, viewpoints, and conclusions
4. Use concise and clear language
5. Appropriately preserve the speaker's expression style and key opinions
6. Output continuous text without deliberate paragraphing or headings
7. If screenshots are provided, consider their visual context when summarizing"""

        default_user = """Based on the following content, write a comprehensive summary in {language_name}:

{transcript}

Requirements:
- Avoid decorative headings
- Cover all key ideas and arguments, preserving important examples and data
- Ensure balanced coverage of both early and later content
- Use restrained but comprehensive language
- Output continuous text without explicit paragraph breaks
- If screenshot references appear in the content, use them as helpful context"""

        system_prompt = self._render_prompt(
            self._get_prompt_value("summarize_single", "system", default=default_system),
            language_name=language_name,
        )
        user_prompt = self._render_prompt(
            self._get_prompt_value("summarize_single", "user", default=default_user),
            language_name=language_name,
            transcript=transcript,
        )

        logger.info(f"正在生成{language_name}摘要...")
        
        # 调用OpenAI API
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if image_payloads:
            content_parts = [{"type": "text", "text": user_prompt}]
            for payload in image_payloads:
                data_uri = payload.get("data_uri")
                if data_uri:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_prompt})

        response = self.summarize_client.chat.completions.create(
            model=self.summarize_model,
            messages=messages,
            max_tokens=3500,  # 控制在安全范围内，避免超出模型限制
            temperature=0.3
        )
        
        summary = response.choices[0].message.content

        return self._format_summary_with_meta(summary, target_language, video_title)

    def _format_summary_with_meta(self, summary: str, target_language: str, video_title: str = None) -> str:
        """
        为摘要添加标题和元信息
        """
        # 不加任何小标题/免责声明，可保留视频标题作为一级标题
        if video_title:
            prefix = f"# {video_title}\n\n"
        else:
            prefix = ""
        return prefix + summary
    
    def _generate_fallback_summary(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        生成备用摘要（当OpenAI API不可用时）
        
        Args:
            transcript: 转录文本
            video_title: 视频标题
            target_language: 目标语言代码
            
        Returns:
            备用摘要文本
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # 简单的文本处理，提取关键信息
        lines = transcript.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('**')]
        
        # 计算大概的长度
        total_chars = sum(len(line) for line in content_lines)
        
        # 使用目标语言的标签
        meta_labels = self._get_summary_labels(target_language)
        fallback_labels = self._get_fallback_labels(target_language)
        
        # 直接使用视频标题作为主标题  
        title = video_title if video_title else "Summary"
        
        summary = f"""# {title}

**{meta_labels['language_label']}:** {language_name}
**{fallback_labels['notice']}:** {fallback_labels['api_unavailable']}



## {fallback_labels['overview_title']}

**{fallback_labels['content_length']}:** {fallback_labels['about']} {total_chars} {fallback_labels['characters']}
**{fallback_labels['paragraph_count']}:** {len(content_lines)} {fallback_labels['paragraphs']}

## {fallback_labels['main_content']}

{fallback_labels['content_description']}

{fallback_labels['suggestions_intro']}

1. {fallback_labels['suggestion_1']}
2. {fallback_labels['suggestion_2']}
3. {fallback_labels['suggestion_3']}

## {fallback_labels['recommendations']}

- {fallback_labels['recommendation_1']}
- {fallback_labels['recommendation_2']}


<br/>

<p style="color: #888; font-style: italic; text-align: center; margin-top: 16px;"><em>{fallback_labels['fallback_disclaimer']}</em></p>"""
        
        return summary
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_supported_languages(self) -> dict:
        """
        获取支持的语言列表
        
        Returns:
            语言代码到语言名称的映射
        """
        return self.language_map.copy()
    
    def _detect_transcript_language(self, transcript: str) -> str:
        """
        检测转录文本的主要语言
        
        Args:
            transcript: 转录文本
            
        Returns:
            检测到的语言代码
        """
        # 简单的语言检测逻辑：查找转录文本中的语言标记
        if "**检测语言:**" in transcript:
            # 从Whisper转录中提取检测到的语言
            lines = transcript.split('\n')
            for line in lines:
                if "**检测语言:**" in line:
                    # 提取语言代码，例如: "**检测语言:** en"
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # 如果没有找到语言标记，使用简单的字符检测
        # 计算英文字符、中文字符等的比例
        total_chars = len(transcript)
        if total_chars == 0:
            return "en"  # 默认英文
            
        # 统计中文字符
        chinese_chars = sum(1 for char in transcript if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars
        
        # 统计英文字母
        english_chars = sum(1 for char in transcript if char.isascii() and char.isalpha())
        english_ratio = english_chars / total_chars
        
        # 根据比例判断
        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "en"  # 默认英文
    

    def _get_summary_labels(self, lang_code: str) -> dict:
        """
        获取摘要页面的多语言标签
        
        Args:
            lang_code: 语言代码
            
        Returns:
            标签字典
        """
        labels = {
            "en": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "zh": {
                "language_label": "摘要语言",
                "disclaimer": "本摘要由AI自动生成，仅供参考"
            },
            "ja": {
                "language_label": "要約言語",
                "disclaimer": "この要約はAIによって自動生成されており、参考用です"
            },
            "ko": {
                "language_label": "요약 언어",
                "disclaimer": "이 요약은 AI에 의해 자동 생성되었으며 참고용입니다"
            },
            "es": {
                "language_label": "Idioma del Resumen",
                "disclaimer": "Este resumen es generado automáticamente por IA, solo para referencia"
            },
            "fr": {
                "language_label": "Langue du Résumé",
                "disclaimer": "Ce résumé est généré automatiquement par IA, à titre de référence uniquement"
            },
            "de": {
                "language_label": "Zusammenfassungssprache",
                "disclaimer": "Diese Zusammenfassung wird automatisch von KI generiert, nur zur Referenz"
            },
            "it": {
                "language_label": "Lingua del Riassunto",
                "disclaimer": "Questo riassunto è generato automaticamente dall'IA, solo per riferimento"
            },
            "pt": {
                "language_label": "Idioma do Resumo",
                "disclaimer": "Este resumo é gerado automaticamente por IA, apenas para referência"
            },
            "ru": {
                "language_label": "Язык резюме",
                "disclaimer": "Это резюме автоматически генерируется ИИ, только для справки"
            },
            "ar": {
                "language_label": "لغة الملخص",
                "disclaimer": "هذا الملخص تم إنشاؤه تلقائياً بواسطة الذكاء الاصطناعي، للمرجع فقط"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def _get_fallback_labels(self, lang_code: str) -> dict:
        """
        获取备用摘要的多语言标签
        
        Args:
            lang_code: 语言代码
            
        Returns:
            标签字典
        """
        labels = {
            "en": {
                "notice": "Notice",
                "api_unavailable": "OpenAI API is unavailable, this is a simplified summary",
                "overview_title": "Transcript Overview",
                "content_length": "Content Length",
                "about": "About",
                "characters": "characters",
                "paragraph_count": "Paragraph Count",
                "paragraphs": "paragraphs",
                "main_content": "Main Content",
                "content_description": "The transcript contains complete video speech content. Since AI summary cannot be generated currently, we recommend:",
                "suggestions_intro": "For detailed information, we suggest you:",
                "suggestion_1": "Review the complete transcript text for detailed information",
                "suggestion_2": "Focus on important paragraphs marked with timestamps",
                "suggestion_3": "Manually extract key points and takeaways",
                "recommendations": "Recommendations",
                "recommendation_1": "Configure OpenAI API key for better summary functionality",
                "recommendation_2": "Or use other AI services for text summarization",
                "fallback_disclaimer": "This is an automatically generated fallback summary"
            },
            "zh": {
                "notice": "注意",
                "api_unavailable": "由于OpenAI API不可用，这是一个简化的摘要",
                "overview_title": "转录概览",
                "content_length": "内容长度",
                "about": "约",
                "characters": "字符",
                "paragraph_count": "段落数量",
                "paragraphs": "段",
                "main_content": "主要内容",
                "content_description": "转录文本包含了完整的视频语音内容。由于当前无法生成智能摘要，建议您：",
                "suggestions_intro": "为获取详细信息，建议您：",
                "suggestion_1": "查看完整的转录文本以获取详细信息",
                "suggestion_2": "关注时间戳标记的重要段落",
                "suggestion_3": "手动提取关键观点和要点",
                "recommendations": "建议",
                "recommendation_1": "配置OpenAI API密钥以获得更好的摘要功能",
                "recommendation_2": "或者使用其他AI服务进行文本总结",
                "fallback_disclaimer": "本摘要为自动生成的备用版本"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def is_openai_configured(self) -> bool:
        """
        检查OpenAI API是否已配置
        
        Returns:
            True if OpenAI API is configured, False otherwise
        """
        return self.optimize_client is not None or self.summarize_client is not None
