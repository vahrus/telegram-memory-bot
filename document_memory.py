import logging
import threading
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from docx import Document
from pypdf import PdfReader

import config
from ai_client import embed_text, embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "telegram_memory"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
_ADD_BATCH_SIZE = 100

_lock = threading.Lock()
_client: Any = None
_collection: Any = None


class DocumentProcessingError(Exception):
    """Ошибка обработки документа, которую можно показать пользователю."""


class RetrievedChunk(TypedDict):
    text: str
    file_name: str
    document_id: str
    chunk_index: int


def user_upload_path(user_id: int, file_name: str) -> Path:
    """Возвращает путь для нового файла в uploads/<user_id>/."""
    config.ensure_storage_dirs()
    user_dir = config.UPLOADS_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / f"{uuid4().hex}_{_safe_file_name(file_name)}"


def extract_text(path: Path) -> str:
    """Извлекает текст из PDF, DOCX или TXT."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix == ".txt":
        return _extract_txt(path)
    raise DocumentProcessingError("Поддерживаются только PDF, DOCX и TXT.")


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Делит текст на фрагменты фиксированной длины с перекрытием."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля.")

    overlap = min(max(chunk_overlap, 0), chunk_size - 1)
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start += step

    return chunks


def save_chunks(
    user_id: int,
    document_id: str,
    file_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    """Сохраняет фрагменты и их embeddings в локальную ChromaDB."""
    if len(chunks) != len(embeddings):
        raise ValueError("Количество фрагментов и embeddings не совпадает.")
    if not chunks:
        raise DocumentProcessingError(
            "Документ пустой или из него не удалось извлечь текст."
        )

    user_key = str(user_id)
    ids = [f"{user_key}:{document_id}:{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "user_id": user_key,
            "document_id": document_id,
            "file_name": file_name,
            "chunk_index": index,
        }
        for index in range(len(chunks))
    ]

    with _lock:
        collection = _get_collection()
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            end = start + _ADD_BATCH_SIZE
            collection.add(
                ids=ids[start:end],
                documents=chunks[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )


def search_relevant_chunks(
    user_id: int,
    query_embedding: list[float],
    top_k: int = TOP_K,
) -> list[RetrievedChunk]:
    """Ищет ближайшие фрагменты только среди документов этого user_id."""
    user_key = str(user_id)

    with _lock:
        collection = _get_collection()
        try:
            stored = collection.get(where={"user_id": user_key}, include=["metadatas"])
        except Exception:
            logger.exception("Не удалось прочитать коллекцию ChromaDB")
            return []

        stored_ids = stored.get("ids") or []
        if not stored_ids:
            return []

        try:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(stored_ids)),
                where={"user_id": user_key},
                include=["documents", "metadatas"],
            )
        except Exception:
            logger.exception("Ошибка семантического поиска в ChromaDB")
            return []

    documents = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    chunks: list[RetrievedChunk] = []

    for document, metadata in zip(documents, metadatas):
        if not document:
            continue
        meta = metadata or {}
        chunks.append(
            {
                "text": document,
                "file_name": str(meta.get("file_name") or ""),
                "document_id": str(meta.get("document_id") or ""),
                "chunk_index": int(meta.get("chunk_index") or 0),
            }
        )

    return chunks


def find_relevant_chunks(user_id: int, question: str) -> list[RetrievedChunk]:
    """Считает embedding вопроса и ищет релевантные фрагменты пользователя."""
    embedding = embed_text(question)
    return search_relevant_chunks(user_id, embedding)


def index_document(user_id: int, file_path: Path, file_name: str) -> int:
    """Читает документ, режет на фрагменты и сохраняет их в долгую память."""
    text = extract_text(file_path)
    chunks = split_text(text)
    if not chunks:
        raise DocumentProcessingError(
            "Документ пустой или из него не удалось извлечь текст."
        )

    embeddings = embed_texts(chunks)
    save_chunks(
        user_id=user_id,
        document_id=uuid4().hex,
        file_name=file_name,
        chunks=chunks,
        embeddings=embeddings,
    )
    logger.info(
        "Документ пользователя %s сохранён в долгую память, фрагментов: %s",
        user_id,
        len(chunks),
    )
    return len(chunks)


def _get_collection() -> Any:
    global _client, _collection
    if _collection is not None:
        return _collection

    config.ensure_storage_dirs()
    _client = chromadb.PersistentClient(
        path=str(config.MEMORY_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _extract_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        logger.exception("Не удалось прочитать PDF «%s»", path.name)
        raise DocumentProcessingError("Не удалось прочитать PDF.") from exc
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    try:
        document = Document(str(path))
    except Exception as exc:
        logger.exception("Не удалось прочитать DOCX «%s»", path.name)
        raise DocumentProcessingError("Не удалось прочитать DOCX.") from exc

    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentProcessingError(
            "Не удалось прочитать TXT: файл должен быть в кодировке UTF-8."
        ) from exc
    except OSError as exc:
        logger.exception("Не удалось прочитать TXT «%s»", path.name)
        raise DocumentProcessingError("Не удалось прочитать документ.") from exc


def _safe_file_name(file_name: str) -> str:
    name = Path(file_name).name
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    cleaned = cleaned.strip("._") or "document"
    return cleaned[:180]
