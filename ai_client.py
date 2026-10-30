import asyncio
import logging
from typing import Any

from openai import OpenAI

from config import CHAT_MODEL, EMBED_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты AI-помощник с короткой и долгой памятью.\n"
    "Используй историю диалога, чтобы понимать контекст беседы.\n"
    "Если предоставлен контекст из документов, используй его как основной источник "
    "фактической информации при вопросах о содержимом документов.\n"
    "Не выдумывай факты, которых нет в документах.\n"
    "Если пользователь спрашивает о документе, но в найденном контексте недостаточно "
    "информации, прямо сообщи, что не удалось найти ответ в загруженных документах.\n"
    "Если вопрос не относится к документам, отвечай как обычный AI-помощник, "
    "учитывая историю разговора.\n"
    "Отвечай понятно, лаконично и на языке пользователя."
)

_EMBED_BATCH_SIZE = 64
_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Возвращает один общий клиент OpenAI на всё приложение."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("Не задана переменная окружения OPENAI_API_KEY.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Создаёт embeddings для списка текстов через OpenAI."""
    if not texts:
        return []

    client = get_openai_client()
    vectors: list[list[float] | None] = [None] * len(texts)

    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for item in response.data:
            vectors[start + item.index] = list(item.embedding)

    if any(vector is None for vector in vectors):
        raise RuntimeError("OpenAI вернул неполный набор embeddings.")

    return [vector for vector in vectors if vector is not None]


def embed_text(text: str) -> list[float]:
    """Создаёт embedding одного текста."""
    vectors = embed_texts([text])
    return vectors[0]


def build_messages(
    question: str,
    history: list[dict[str, str]],
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Собирает запрос: system, короткая история, контекст документа, текущий вопрос.

    Текущий вопрос берётся только из аргумента question. В history его быть не должно.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "system", "content": _format_document_context(chunks)})
    messages.append({"role": "user", "content": question})
    return messages


def _format_document_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return (
            "Контекст из документов: релевантные фрагменты не найдены. "
            "Если вопрос касается загруженных документов, прямо сообщи, что не удалось "
            "найти ответ в загруженных документах. "
            "Если вопрос не про документы, ответь как обычный помощник."
        )

    blocks: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        file_name = str(chunk.get("file_name") or "документ")
        text = str(chunk.get("text") or "").strip()
        blocks.append(f"[{number}] Файл: {file_name}\n{text}")

    return "Контекст из документов пользователя:\n\n" + "\n\n".join(blocks)


def _request_completion(messages: list[dict[str, str]]) -> str:
    response = get_openai_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("Модель вернула пустой ответ.")
    return content.strip()


async def generate_answer(
    question: str,
    history: list[dict[str, str]],
    chunks: list[dict[str, Any]],
) -> str:
    """Отправляет вопрос, короткую историю и фрагменты документа в OpenAI."""
    messages = build_messages(question, history, chunks)
    logger.info(
        "Запрос к модели %s: сообщений в истории=%s, фрагментов документа=%s",
        CHAT_MODEL,
        len(history),
        len(chunks),
    )
    return await asyncio.to_thread(_request_completion, messages)
