import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"
UPLOADS_DIR = BASE_DIR / "uploads"

load_dotenv(BASE_DIR / ".env")


def _read_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


BOT_TOKEN = _read_env("BOT_TOKEN")
OPENAI_API_KEY = _read_env("OPENAI_API_KEY")
CHAT_MODEL = _read_env("CHAT_MODEL", "gpt-5-mini") or "gpt-5-mini"
EMBED_MODEL = _read_env("EMBED_MODEL", "text-embedding-3-small") or "text-embedding-3-small"


class ConfigError(RuntimeError):
    """Не хватает обязательных переменных окружения."""


def validate_config() -> None:
    """Проверяет, что обязательные ключи заданы. Сами ключи никуда не выводит."""
    missing = [
        name
        for name in ("BOT_TOKEN", "OPENAI_API_KEY")
        if not _read_env(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise ConfigError(
            "Не заданы обязательные переменные окружения: "
            f"{joined}. Скопируйте .env.example в .env и заполните значения."
        )


def ensure_storage_dirs() -> None:
    """Создаёт каталоги долгой памяти и загрузок, если их ещё нет."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
