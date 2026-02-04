import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH_JSONC = Path(__file__).resolve().parent.parent / "config.jsonc"


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH_JSONC
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonc":
            text = _strip_jsonc_comments(text)
        return json.loads(text)
    except Exception as exc:
        logger.warning("Failed to read config file %s: %s", path, exc)
        return {}


def _strip_jsonc_comments(content: str) -> str:
    """Remove // comments from JSONC while preserving string literals."""
    lines = []
    for line in content.splitlines():
        in_string = False
        escaped = False
        result = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and not escaped:
                escaped = True
                result.append(ch)
                i += 1
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
                result.append(ch)
                i += 1
                continue
            if not in_string and ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            result.append(ch)
            escaped = False
            i += 1
        lines.append("".join(result).rstrip())
    return "\n".join(lines)


def _set_env_if_value(key: str, value: Optional[str], override: bool) -> None:
    if value is None:
        return
    value_str = str(value).strip()
    if not value_str:
        return
    if override or not os.getenv(key):
        os.environ[key] = value_str


def apply_config_to_env(config: Dict[str, Any], override: bool = False) -> None:
    if not isinstance(config, dict):
        return
    openai_cfg = config.get("openai", {})
    if not isinstance(openai_cfg, dict):
        return

    default_cfg = openai_cfg.get("default", {})
    optimize_cfg = openai_cfg.get("optimize", {})
    summarize_cfg = openai_cfg.get("summarize", {})

    if isinstance(default_cfg, dict):
        _set_env_if_value("OPENAI_API_KEY", default_cfg.get("api_key"), override)
        _set_env_if_value("OPENAI_BASE_URL", default_cfg.get("base_url"), override)

    if isinstance(optimize_cfg, dict):
        _set_env_if_value("OPENAI_OPTIMIZE_API_KEY", optimize_cfg.get("api_key"), override)
        _set_env_if_value("OPENAI_OPTIMIZE_BASE_URL", optimize_cfg.get("base_url"), override)
        _set_env_if_value("OPENAI_OPTIMIZE_MODEL", optimize_cfg.get("model"), override)

    if isinstance(summarize_cfg, dict):
        _set_env_if_value("OPENAI_SUMMARIZE_API_KEY", summarize_cfg.get("api_key"), override)
        _set_env_if_value("OPENAI_SUMMARIZE_BASE_URL", summarize_cfg.get("base_url"), override)
        _set_env_if_value("OPENAI_SUMMARIZE_MODEL", summarize_cfg.get("model"), override)
