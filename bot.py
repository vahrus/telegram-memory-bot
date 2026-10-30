import asyncio
import logging
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from openai import APIError

from ai_client import generate_answer
from config import BOT_TOKEN, CHAT_MODEL, ConfigError, ensure_storage_dirs, validate_config
from document_memory import (
    DocumentProcessingError,
    find_relevant_chunks,
    index_document,
    user_upload_path,
)
from short_memory import add_message, clear_history, get_history

logger = logging.getLogger(__name__)
router = Router()

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}
TELEGRAM_MESSAGE_LIMIT = 4000

START_TEXT = (
    "Привет! Я AI-бот с короткой и долгой памятью.\n\n"
    "🧠 Я помню последние сообщения нашего диалога.\n"
    "📄 Можешь загрузить PDF, DOCX или TXT — я сохраню информацию "
    "и смогу отвечать на вопросы по документу.\n\n"
    "Команды:\n"
    "/reset — очистить историю диалога\n"
    "/help — помощь"
)

HELP_TEXT = (
    "Как устроена память и как задавать вопросы.\n\n"
    "🧠 Короткая память\n"
    "Я помню последние 10 сообщений нашего диалога и использую их, "
    "чтобы отвечать связно. История отдельная для каждого пользователя. "
    "Команда /reset очищает только её. Документы при этом остаются.\n\n"
    "📄 Как загрузить документ\n"
    "Отправь файл PDF, DOCX или TXT обычным сообщением. "
    "Я извлеку текст, разобью его на фрагменты и сохраню в долгой памяти.\n\n"
    "❓ Как задавать вопросы\n"
    "Напиши вопрос обычным текстом. Если документы уже загружены, "
    "я найду подходящие фрагменты и отвечу по ним. "
    "Если документов нет, отвечу как обычный помощник и буду опираться на диалог.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/reset — очистить историю диалога\n"
    "/help — эта справка"
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("reset"))
async def cmd_reset(message: Message) -> None:
    user = message.from_user
    if user is None:
        return
    clear_history(user.id)
    await message.answer("🧹 История текущего диалога очищена.")


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    user = message.from_user
    document = message.document
    if user is None or document is None:
        return

    file_name = Path(document.file_name or "document").name
    if Path(file_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        await message.answer("❌ Поддерживаются только PDF, DOCX и TXT.")
        return

    destination = user_upload_path(user.id, file_name)
    try:
        await bot.download(document, destination=destination, timeout=120)
    except TelegramAPIError:
        logger.exception("Не удалось скачать файл из Telegram")
        _remove_file(destination)
        await message.answer("❌ Не удалось скачать файл из Telegram. Попробуй ещё раз.")
        return
    except Exception:
        logger.exception("Ошибка при сохранении файла")
        _remove_file(destination)
        await message.answer("❌ Не удалось сохранить файл. Попробуй ещё раз.")
        return

    await message.answer("📄 Читаю документ и сохраняю его в долгую память...")
    try:
        chunk_count = await asyncio.to_thread(
            index_document,
            user.id,
            destination,
            file_name,
        )
    except DocumentProcessingError as exc:
        logger.warning("Документ не обработан: %s", exc)
        _remove_file(destination)
        await message.answer(f"❌ {exc}")
        return
    except APIError:
        logger.exception("Ошибка OpenAI при индексации документа")
        _remove_file(destination)
        await message.answer(
            "❌ Не удалось создать embeddings. Проверь доступ к OpenAI API и попробуй ещё раз."
        )
        return
    except Exception:
        logger.exception("Ошибка при обработке документа")
        _remove_file(destination)
        await message.answer("❌ Не удалось обработать документ. Попробуй другой файл.")
        return

    await message.answer(
        f"✅ Документ «{file_name}» обработан.\n\n"
        f"Создано фрагментов: {chunk_count}.\n"
        "Теперь можешь задавать вопросы по документу."
    )


@router.message(F.text)
async def handle_question(message: Message) -> None:
    user = message.from_user
    if user is None or not message.text:
        return

    question = message.text.strip()
    if not question or question.startswith("/"):
        return

    user_id = user.id
    # История берётся до ответа. Текущий вопрос попадает в память только после
    # успешной отправки, поэтому в запрос к модели он входит один раз.
    history = get_history(user_id)
    try:
        chunks = await asyncio.to_thread(find_relevant_chunks, user_id, question)
        answer = await generate_answer(question, history, chunks)
    except APIError:
        logger.exception("Ошибка OpenAI при ответе пользователю %s", user_id)
        await message.answer("❌ Сервис OpenAI сейчас недоступен. Попробуй ещё раз позже.")
        return
    except Exception:
        logger.exception("Не удалось ответить пользователю %s", user_id)
        await message.answer("❌ Не удалось получить ответ. Попробуй ещё раз позже.")
        return

    try:
        await _send_text(message, answer)
    except TelegramAPIError:
        logger.exception("Не удалось отправить ответ пользователю %s", user_id)
        return

    add_message(user_id, "user", question)
    add_message(user_id, "assistant", answer)


async def _send_text(message: Message, text: str) -> None:
    for start in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        await message.answer(text[start : start + TELEGRAM_MESSAGE_LIMIT])


def _remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Не удалось удалить временный файл %s", path.name)


async def main() -> None:
    validate_config()
    ensure_storage_dirs()
    bot = Bot(token=BOT_TOKEN)
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    logger.info("Бот запущен. Модель чата: %s", CHAT_MODEL)
    try:
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for noisy_logger in ("httpx", "httpcore", "httpcore2", "openai", "chromadb"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    try:
        asyncio.run(main())
    except ConfigError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
    except Exception:
        logger.exception("Бот завершился с ошибкой")
        raise SystemExit(1) from None


if __name__ == "__main__":
    run()
