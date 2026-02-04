import logging
from openai import OpenAI
from typing import Any, Optional
import re

from config_loader import load_config

logger = logging.getLogger(__name__)

class Translator:
    """文本翻译器，使用GPT-4o进行高质量翻译"""
    
    def __init__(self):
        self.client = None
        self._init_openai_client()

        config = load_config()
        self.prompt_config = {}
        if isinstance(config, dict):
            prompts = config.get("prompts", {})
            if isinstance(prompts, dict):
                self.prompt_config = prompts.get("translator", {}) or {}
        
        # 语言映射
        self.language_map = {
            "zh": "中文（简体）",
            "zh-tw": "中文（繁体）", 
            "en": "English",
            "ja": "日本語",
            "ko": "한국어",
            "fr": "Français",
            "de": "Deutsch",
            "es": "Español",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ar": "العربية",
            "hi": "हिन्दी"
        }
    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            
            if not api_key:
                logger.warning("未设置OPENAI_API_KEY环境变量")
                return
                
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            logger.info("OpenAI客户端初始化成功")
            
        except Exception as e:
            logger.error(f"初始化OpenAI客户端失败: {str(e)}")
            self.client = None

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
    
    def _detect_source_language(self, text: str) -> str:
        """检测源文本语言"""
        # 简单的语言检测逻辑
        if "**检测语言:**" in text:
            lines = text.split('\n')
            for line in lines:
                if "**检测语言:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # 基于字符统计的简单检测
        total_chars = len(text)
        if total_chars == 0:
            return "en"
        
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        chinese_ratio = chinese_chars / total_chars
        
        # 统计日文字符
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        japanese_ratio = japanese_chars / total_chars
        
        # 统计韩文字符
        korean_chars = len(re.findall(r'[\uac00-\ud7af]', text))
        korean_ratio = korean_chars / total_chars
        
        if chinese_ratio > 0.1:
            return "zh"
        elif japanese_ratio > 0.05:
            return "ja"
        elif korean_ratio > 0.05:
            return "ko"
        else:
            return "en"

    def _normalize_lang_code(self, lang: Optional[str]) -> Optional[str]:
        if not lang:
            return None
        code = lang.lower().strip()
        code = code.replace("*", "").replace("`", "").strip()
        if code.startswith("zh"):
            return "zh"
        if code.startswith("en"):
            return "en"
        if "-" in code:
            return code.split("-", 1)[0]
        return code
    
    async def translate_text(self, text: str, target_language: str, source_language: Optional[str] = None) -> str:
        """
        翻译文本到目标语言
        
        Args:
            text: 要翻译的文本
            target_language: 目标语言代码
            source_language: 源语言代码（可选，会自动检测）
            
        Returns:
            翻译后的文本
        """
        try:
            if not self.client:
                logger.warning("OpenAI API不可用，无法翻译")
                return text
            
            # 检测源语言
            if not source_language:
                source_language = self._detect_source_language(text)

            normalized_source = self._normalize_lang_code(source_language)
            normalized_target = self._normalize_lang_code(target_language)

            # 如果源语言和目标语言相同，直接返回
            if normalized_source and normalized_target and normalized_source == normalized_target:
                return text

            source_lang_key = normalized_source or source_language
            target_lang_key = normalized_target or target_language
            source_lang_name = self.language_map.get(source_lang_key, source_lang_key)
            target_lang_name = self.language_map.get(target_lang_key, target_lang_key)
            
            logger.info(f"开始翻译：{source_lang_name} -> {target_lang_name}")
            
            return await self._translate_single_text(text, target_lang_name, source_lang_name)
                
        except Exception as e:
            logger.error(f"翻译失败: {str(e)}")
            return text
    
    async def _translate_single_text(self, text: str, target_lang_name: str, source_lang_name: str) -> str:
        """翻译单个文本块"""
        default_system = """你是专业翻译专家。请将{source_lang_name}文本准确翻译为{target_lang_name}。

翻译要求：
- 保持原文的格式和结构（包括段落分隔、标题等）
- 准确传达原意，语言自然流畅
- 保留专业术语的准确性
- 不要添加解释或注释
- 如果遇到Markdown格式，请保持格式不变"""

        default_user = """请将以下{source_lang_name}文本翻译为{target_lang_name}：

{text}

只返回翻译结果，不要添加任何说明。"""

        system_prompt = self._render_prompt(
            self._get_prompt_value("translate", "system", default=default_system),
            source_lang_name=source_lang_name,
            target_lang_name=target_lang_name,
        )
        user_prompt = self._render_prompt(
            self._get_prompt_value("translate", "user", default=default_user),
            source_lang_name=source_lang_name,
            target_lang_name=target_lang_name,
            text=text,
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"单文本翻译失败: {e}")
            return text
    
    def should_translate(self, source_language: str, target_language: str) -> bool:
        """判断是否需要翻译"""
        if not source_language or not target_language:
            return False
        
        # 标准化语言代码
        source_lang = self._normalize_lang_code(source_language)
        target_lang = self._normalize_lang_code(target_language)
        if not source_lang or not target_lang:
            return False
        
        # 如果语言相同，不需要翻译
        if source_lang == target_lang:
            return False
        
        # 处理中文的特殊情况
        chinese_variants = ["zh", "zh-cn", "zh-hans", "chinese"]
        if source_lang in chinese_variants and target_lang in chinese_variants:
            return False
        
        return True
