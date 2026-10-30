from collections import defaultdict, deque
from typing import Literal, TypedDict

Role = Literal["user", "assistant"]
HISTORY_LIMIT = 10


class ChatMessage(TypedDict):
    role: Role
    content: str


_history: dict[int, deque[ChatMessage]] = defaultdict(
    lambda: deque(maxlen=HISTORY_LIMIT)
)


def get_history(user_id: int) -> list[ChatMessage]:
    """Возвращает копию последних сообщений пользователя."""
    history = _history.get(user_id)
    if not history:
        return []
    return [{"role": item["role"], "content": item["content"]} for item in history]


def add_message(user_id: int, role: Role, content: str) -> None:
    """Добавляет одно сообщение в короткую память конкретного пользователя."""
    _history[user_id].append({"role": role, "content": content})


def clear_history(user_id: int) -> None:
    """Очищает только короткую память. Документы в ChromaDB не затрагиваются."""
    if user_id in _history:
        _history[user_id].clear()
