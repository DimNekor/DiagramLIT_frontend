import reflex as rx
from pydantic import BaseModel
import asyncio
import base64
import httpx
import json
import os
import re
import time
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

LOCAL_VLM_URL = os.getenv("LOCAL_VLM_URL", "http://localhost:2023")
ORANGEPI_URL = os.getenv("ORANGEPI_URL", "http://localhost:2022")

# Таймаут на submit (CV: детекция + OCR — секунды/десятки секунд)
ORANGEPI_CV_TIMEOUT = float(os.getenv("ORANGEPI_CV_TIMEOUT", 120.0))
# Таймаут на один poll-запрос (лёгкий, должен отвечать мгновенно)
ORANGEPI_POLL_TIMEOUT = float(os.getenv("ORANGEPI_POLL_TIMEOUT", 30.0))
# Интервал опроса статуса LLM
ORANGEPI_POLL_INTERVAL = float(os.getenv("ORANGEPI_POLL_INTERVAL", 2.0))

BPMN_ANALYSIS_PROMPT = """
Ты — эксперт по анализу BPMN-диаграмм и информационной безопасности.

Проанализируй предоставленное изображение BPMN-диаграммы и верни ответ СТРОГО в виде JSON-объекта без markdown-разметки (без ```json), без пояснений, только сырой JSON.

Для распознавания элементов используй СТРОГО следующий список классов (как в модели детекции):
- arrow_end (конец стрелки/поток управления)
- data_base (база данных/хранилище)
- data_object (объект данных)
- exclusive_gateway (эксклюзивный шлюз, XOR)
- finish_event (конечное событие)
- inclusive_gateway (инклюзивный шлюз, OR)
- intermediate_event (промежуточное событие)
- parallel_gateway (параллельный шлюз, AND)
- role (название процесса ИЛИ роль участника. Это заголовки пулов и дорожек. Рамка должна охватывать весь прямоугольник заголовка вместе с текстом)
- start_event (стартовое событие)
- task (задача/действие)

Структура ответа:
{
  "diagram_type": "<тип диаграммы: BPMN Process Diagram / Collaboration / Choreography / другое>",
  "detected_elements": [
    "<класс_элемента: текст внутри, например: 'start_event: Запрос получен', 'task: Оплата', 'role: Бухгалтерия'>",
    "..."
  ],
  "relationships": [
    "<связь в формате 'Источник → Цель', например: 'start_event → task: Оплата'>",
    "..."
  ],
  "bounding_boxes": [
    {
      "class": "<строго одно из названий классов из списка выше>",
      "label": "<распознанный текст внутри элемента, если есть>",
      "x": 0.0,
      "y": 0.0,
      "w": 0.1,
      "h": 0.1
    }
  ],
  "step_by_step_description": [
    "<шаг 1: подробное описание>",
    "..."
  ],
  "security_issues": [
    "<проблема безопасности 1>",
    "..."
  ]
}

Правила для bounding_boxes:
- x, y — нормализованные координаты левого верхнего угла элемента (0.0–1.0)
- w, h — нормализованные ширина и высота (доля от размеров изображения)
- Укажи bounding box для каждого элемента из detected_elements.
- ВАЖНО для класса 'role': этот класс является общим для названий ролей (дорожек) и названий процессов. Bounding box для 'role' должен строго обводить прямоугольный блок (заголовок пула/дорожки), в котором написан соответствующий текст.

Правила для security_issues:
- Перечисли архитектурные проблемы безопасности, видимые в диаграмме: отсутствие аутентификации/авторизации, незащищённые потоки данных, отсутствие обработки ошибок, потенциальные DoS-векторы, нарушения принципа наименьших привилегий и т.д.
- Если явных проблем нет — верни пустой список [].

Верни ТОЛЬКО JSON-объект.
"""


def _parse_model_json(text: str) -> Dict[str, Any]:
    """Извлекает JSON из ответа модели, устойчив к markdown-обёрткам."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Не удалось разобрать JSON из ответа модели: {text[:300]}")


# ──────────────────────────────────────────────────────────────────────────
#  Конфигурация доступных бэкендов (устройство + модель)
# ──────────────────────────────────────────────────────────────────────────
MODEL_BACKENDS: Dict[str, str] = {
    "gemini": "Gemini 3.1 Pro · облако",
    "local_vlm": "Локальная VLM · 8 ГБ VRAM · RU",
    "orangepi": "YOLOv8 + OCR · OrangePi RV2",
}


class BBox(BaseModel):
    """Bounding box одного распознанного элемента.
    Координаты — ДОЛИ от размеров изображения (0..1), а не пиксели."""

    label: str
    x: float
    y: float
    w: float
    h: float


def _demo_result(diagram_type: str = "BPMN Process Diagram") -> Dict[str, Any]:
    """Временная заглушка для неизвестных моделей."""
    return {
        "diagram_type": diagram_type,
        "detected_elements": ["Start Event", "Task: Проверка заказа", "End Event"],
        "relationships": ["Start → Проверка заказа", "Проверка заказа → End"],
        "bounding_boxes": [],
        "step_by_step_description": ["Демо-режим: модель не подключена."],
        "security_issues": [],
    }


# ──────────────────────────────────────────────────────────────────────────
#  Функции инференса (вне State — им не нужен доступ к состоянию)
# ──────────────────────────────────────────────────────────────────────────
async def _infer_gemini(image_bytes: bytes) -> Dict[str, Any]:
    """Инференс через Gemini via OpenAI-compatible Cloudflare Worker.
    Воркер возвращает SSE-стрим независимо от запроса, поэтому stream=True."""
    client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    stream = await client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BPMN_ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        stream=True,
    )
    text = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
    print(f"[DiagramLIT] Gemini raw response ({len(text)} chars): {text[:300]!r}")
    stripped = text.strip()
    if not stripped.startswith("{"):
        if "429" in stripped:
            raise RuntimeError(
                "Превышен лимит запросов к Gemini API (429). "
                "Подождите 30–60 секунд и попробуйте снова."
            )
        raise RuntimeError(f"Неожиданный ответ от API: {stripped[:200]}")
    return _parse_model_json(text)


async def _infer_local_vlm(image_bytes: bytes) -> Dict[str, Any]:
    raise NotImplementedError(
        f"Локальная VLM ещё не подключена. Запустите FastAPI-бэкенд на {LOCAL_VLM_URL}."
    )


async def _orangepi_submit(image_bytes: bytes) -> Dict[str, Any]:
    """Сабмит на OrangePi: POST /infer. Возвращает CV-результат + job_id СРАЗУ.
    Это быстрый запрос (только детекция и OCR), без ожидания LLM."""
    base_url = ORANGEPI_URL.rstrip("/")
    url = f"{base_url}/infer"
    files = {"file": ("diagram.png", image_bytes, "image/png")}
    timeout = httpx.Timeout(ORANGEPI_CV_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, files=files)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Не удалось подключиться к OrangePi по адресу {url}. "
                "Убедитесь, что FastAPI запущен, а обратный SSH-туннель на порту 2022 активен."
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Таймаут детекции: OrangePi не ответил за {int(ORANGEPI_CV_TIMEOUT)} секунд."
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ошибка на стороне FastAPI (код {e.response.status_code}): {e.response.text}"
            )


async def _orangepi_poll(job_id: str) -> Dict[str, Any]:
    """Опрос статуса джобы: GET /infer/{job_id}. Лёгкий короткий запрос."""
    base_url = ORANGEPI_URL.rstrip("/")
    url = f"{base_url}/infer/{job_id}"
    timeout = httpx.Timeout(ORANGEPI_POLL_TIMEOUT, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


# Только синхронные «один ответ целиком» модели. OrangePi обрабатывается
# отдельной веткой (submit + poll), поэтому его здесь нет.
INFER_FUNCS = {
    "gemini": _infer_gemini,
    "local_vlm": _infer_local_vlm,
}


class State(rx.State):
    """Состояние приложения DiagramLIT"""

    is_uploading: bool = False
    retry_message: str = ""
    progress_message: str = ""
    cancel_requested: bool = False

    # Статус фоновой генерации описания на OrangePi:
    # "" (не используется) / "pending" / "done" / "error"
    llm_status: str = ""
    llm_job_id: str = ""

    selected_model: str = "gemini"
    image_data_url: str = ""

    uploaded_filename: str = ""
    diagram_type: str = ""
    detected_elements: List[str] = []
    relationships: List[str] = []
    step_by_step_description: List[str] = []
    bounding_boxes: List[BBox] = []
    security_issues: List[str] = []
    error_message: str = ""

    @rx.var
    def has_error(self) -> bool:
        return self.diagram_type == "Ошибка" and bool(self.error_message)

    @rx.var
    def model_label(self) -> str:
        return MODEL_BACKENDS.get(self.selected_model, self.selected_model)

    @rx.var
    def steps_text(self) -> str:
        return "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(self.step_by_step_description)
        )

    def set_selected_model(self, value: str):
        self.selected_model = value

    def _reset_results(self):
        self.diagram_type = ""
        self.detected_elements = []
        self.relationships = []
        self.step_by_step_description = []
        self.bounding_boxes = []
        self.security_issues = []
        self.error_message = ""
        self.retry_message = ""
        self.progress_message = ""
        self.llm_status = ""
        self.llm_job_id = ""
        self.image_data_url = ""
        self.cancel_requested = False

    def _apply_cv_result(self, result: Dict[str, Any]):
        """Раскладывает результат (CV или полный) по полям состояния.
        Вызывать ТОЛЬКО внутри `async with self` (мутирует state)."""
        self.diagram_type = result.get("diagram_type", "Неизвестный тип")
        self.detected_elements = result.get("detected_elements", [])
        self.relationships = result.get("relationships", [])
        self.step_by_step_description = result.get("step_by_step_description", [])
        self.security_issues = result.get("security_issues", [])

        boxes: List[BBox] = []
        for b in result.get("bounding_boxes", []):
            try:
                boxes.append(
                    BBox(
                        label=str(b.get("label", "")),
                        x=float(b.get("x", 0)),
                        y=float(b.get("y", 0)),
                        w=float(b.get("w", 0)),
                        h=float(b.get("h", 0)),
                    )
                )
            except Exception:
                pass
        self.bounding_boxes = boxes

    # ── Шаг 1. Быстрый обработчик загрузки: только принять файл ──────────
    async def handle_upload(self, files: List[rx.UploadFile]):
        """Включает спиннер, читает файл и СРАЗУ завершается, передавая
        тяжёлую работу фоновой задаче run_analysis. Никакого инференса здесь!"""
        if not files:
            self.is_uploading = False
            return
        self._reset_results()
        self.is_uploading = True
        self.uploaded_filename = files[0].filename
        yield  # отправляем спиннер клиенту до чтения файла
        file_content = await files[0].read()
        self.image_data_url = (
            "data:image/png;base64," + base64.b64encode(file_content).decode()
        )
        yield State.run_analysis

    # ── Инференс одной попытки с heartbeat (держит websocket живым) ──────
    async def _infer_with_heartbeat(self, infer, image_bytes: bytes) -> Dict[str, Any]:
        """Запускает infer(image_bytes) задачей и каждые несколько секунд
        обновляет state — этот трафик по websocket не даёт соединению уснуть
        на долгих запросах. Возвращает результат или пробрасывает исключение."""
        task = asyncio.create_task(infer(image_bytes))
        start = time.monotonic()
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=5.0)
                async with self:
                    if self.cancel_requested:
                        task.cancel()
                        raise asyncio.CancelledError()
                    elapsed = int(time.monotonic() - start)
                    self.progress_message = f"Идёт обработка на устройстве… {elapsed // 60}:{elapsed % 60:02d}"
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise
        return task.result()

    # ── Шаг 2. Тяжёлый инференс в фоне ────────────────────────────────────
    @rx.event(background=True)
    async def run_analysis(self):
        """Фоновая задача. Для OrangePi: submit (CV сразу) → redirect →
        poll (описание подъезжает). Для облачных моделей: один ответ целиком."""
        try:
            async with self:
                if not self.image_data_url:
                    self.is_uploading = False
                    return
                image_bytes = base64.b64decode(self.image_data_url.split(",", 1)[1])
                model_key = self.selected_model

            # ============ OrangePi: submit (CV) + poll (LLM) ============
            if model_key == "orangepi":
                # 1) Сабмит — быстрый POST. CV-результат и job_id приходят сразу.
                #    Оборачиваем в heartbeat: OCR может занять десятки секунд.
                try:
                    submit = await self._infer_with_heartbeat(
                        _orangepi_submit, image_bytes
                    )
                except asyncio.CancelledError:
                    async with self:
                        self.is_uploading = False
                        self.retry_message = ""
                        self.progress_message = ""
                    return
                except Exception as e:
                    async with self:
                        self.is_uploading = False
                        self.retry_message = ""
                        self.progress_message = ""
                        self.error_message = str(e)
                        self.diagram_type = "Ошибка"
                    yield rx.redirect("/results")
                    return

                # Бэкенд при провале детекции возвращает diagram_type="Ошибка"
                if submit.get("diagram_type") == "Ошибка":
                    async with self:
                        self.is_uploading = False
                        self.progress_message = ""
                        self.error_message = submit.get(
                            "error_message", "Ошибка детекции"
                        )
                        self.diagram_type = "Ошибка"
                    yield rx.redirect("/results")
                    return

                job_id = submit.get("job_id", "")
                async with self:
                    if self.cancel_requested:
                        self.is_uploading = False
                        self.progress_message = ""
                        return
                    self._apply_cv_result(submit)
                    self.llm_job_id = job_id
                    self.llm_status = "pending" if job_id else "done"
                    self.progress_message = "ИИ готовит описание…" if job_id else ""
                    self.is_uploading = False
                # CV готов — показываем результаты немедленно
                yield rx.redirect("/results")

                if not job_id:
                    return

                # 2) Поллинг — короткие GET-запросы, пока описание не готово.
                start = time.monotonic()
                while True:
                    await asyncio.sleep(ORANGEPI_POLL_INTERVAL)
                    async with self:
                        if self.cancel_requested:
                            self.llm_status = ""
                            self.progress_message = ""
                            return
                    try:
                        poll = await _orangepi_poll(job_id)
                    except Exception:
                        # сеть/туннель моргнули — повторим на следующей итерации
                        continue

                    status = poll.get("status", "pending")
                    if status in ("done", "error"):
                        async with self:
                            self.step_by_step_description = poll.get(
                                "step_by_step_description",
                                self.step_by_step_description,
                            )
                            self.llm_status = status
                            self.progress_message = ""
                        return

                    elapsed = int(time.monotonic() - start)
                    async with self:
                        self.progress_message = (
                            f"ИИ готовит описание… {elapsed // 60}:{elapsed % 60:02d}"
                        )

            # ============ Облачные/прочие модели: один ответ целиком ============
            else:
                infer = INFER_FUNCS.get(model_key)
                result: Dict[str, Any] | None = None
                error: Exception | None = None

                for attempt in range(3):
                    try:
                        if infer is None:
                            result = _demo_result("Неизвестная модель")
                        else:
                            result = await self._infer_with_heartbeat(
                                infer, image_bytes
                            )
                        error = None
                        break
                    except asyncio.CancelledError:
                        async with self:
                            self.is_uploading = False
                            self.retry_message = ""
                            self.progress_message = ""
                        return
                    except Exception as e:
                        error = e
                        if "429" not in str(e) or attempt >= 2:
                            break
                        wait = 35
                        async with self:
                            if self.cancel_requested:
                                return
                            self.progress_message = ""
                            self.retry_message = (
                                f"Лимит запросов API. "
                                f"Повтор {attempt + 1}/2 через {wait} сек..."
                            )
                        await asyncio.sleep(wait)
                        async with self:
                            self.retry_message = ""
                            if self.cancel_requested:
                                return

                async with self:
                    if self.cancel_requested:
                        self.is_uploading = False
                        self.retry_message = ""
                        self.progress_message = ""
                        return
                    self.is_uploading = False
                    self.retry_message = ""
                    self.progress_message = ""
                    if error is not None or result is None:
                        self.error_message = (
                            str(error) if error else "Неизвестная ошибка"
                        )
                        self.diagram_type = "Ошибка"
                        print(f"[DiagramLIT] Ошибка: {error}")
                    else:
                        self._apply_cv_result(result)
                        self.llm_status = (
                            ""  # для облачных моделей отдельной LLM-фазы нет
                        )
                yield rx.redirect("/results")

        except Exception as e:
            # Страховка: что бы ни случилось, спиннер НЕ зависнет.
            traceback.print_exc()
            async with self:
                self.is_uploading = False
                self.retry_message = ""
                self.progress_message = ""
                self.error_message = str(e)
                self.diagram_type = "Ошибка"
            yield rx.redirect("/results")

    # ── Аварийный выход из спиннера ───────────────────────────────────────
    def cancel_analysis(self):
        """Кнопка «Отменить»: мгновенно возвращает форму, а фоновая задача
        увидит флаг и тихо завершится без redirect."""
        self.cancel_requested = True
        self.is_uploading = False
        self.retry_message = ""
        self.progress_message = ""
        self.llm_status = ""

    def clear_upload(self):
        self.uploaded_filename = ""
        self._reset_results()


def navbar():
    return rx.hstack(
        rx.hstack(
            rx.image(src="/favicon.ico", width="32px", height="32px"),
            rx.heading("DiagramLIT", size="6", color="white", font_weight="bold"),
            spacing="2",
            align="center",
        ),
        rx.hstack(
            rx.link("Главная", href="/", color="white", font_weight="500"),
            rx.link(
                "GitHub",
                href="https://github.com",
                color="white",
                font_weight="500",
                is_external=True,
            ),
            spacing="4",
        ),
        justify="between",
        align="center",
        padding_x="2em",
        padding_y="1em",
        bg="rgba(255, 255, 255, 0.1)",
        backdrop_filter="blur(10px)",
        box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1)",
        width="100%",
    )


def footer():
    return rx.center(
        rx.vstack(
            rx.text(
                "© 2026 DiagramLIT — AI-powered Diagram Analysis",
                font_size="0.8em",
                color="#a0aec0",
            ),
            rx.text(
                "Распознавание BPMN-диаграмм",
                font_size="0.7em",
                color="#a0aec0",
            ),
            align="center",
            spacing="1",
        ),
        padding="2em",
        bg="rgba(0, 0, 0, 0.3)",
        width="100%",
    )


def model_selector() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.icon("cpu", size=20, color="#667eea"),
            rx.text(
                "Устройство и модель",
                font_size="0.95em",
                font_weight="600",
                color="#1a202c",
            ),
            spacing="2",
            align="center",
        ),
        rx.select.root(
            rx.select.trigger(placeholder="Выберите модель", width="100%"),
            rx.select.content(
                rx.select.group(
                    rx.select.item(MODEL_BACKENDS["gemini"], value="gemini"),
                    rx.select.item(MODEL_BACKENDS["local_vlm"], value="local_vlm"),
                    rx.select.item(MODEL_BACKENDS["orangepi"], value="orangepi"),
                ),
            ),
            value=State.selected_model,
            on_change=State.set_selected_model,
            width="100%",
        ),
        spacing="2",
        align="start",
        width="100%",
    )


def loading_content() -> rx.Component:
    """Анимированный контент внутри карточки, пока идёт инференс."""
    return rx.vstack(
        rx.html(
            "<style>"
            "@keyframes dl-spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}"
            "@keyframes dl-fadein{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}"
            "</style>"
        ),
        rx.box(
            style={
                "width": "72px",
                "height": "72px",
                "border": "5px solid #e2e8f0",
                "border-top-color": "#667eea",
                "border-radius": "50%",
                "animation": "dl-spin 1s linear infinite",
                "margin": "0 auto",
            },
        ),
        rx.vstack(
            rx.heading("Анализируем диаграмму", size="5", color="#1a202c"),
            rx.text(
                f"Модель: {State.model_label}",
                font_size="0.85em",
                color="#718096",
            ),
            spacing="1",
            align="center",
        ),
        rx.divider(),
        rx.vstack(
            rx.hstack(
                rx.text(
                    "✓",
                    color="#38a169",
                    font_weight="700",
                    font_size="1.1em",
                    min_width="1.8em",
                ),
                rx.text("Изображение загружено", font_size="0.95em", color="#2d3748"),
                align="center",
                width="100%",
                style={
                    "animation": "dl-fadein 0.5s ease 0.3s forwards",
                    "opacity": "0",
                },
            ),
            rx.hstack(
                rx.spinner(size="1", color="#667eea"),
                rx.text("Отправляем в модель...", font_size="0.95em", color="#2d3748"),
                spacing="3",
                align="center",
                width="100%",
                style={
                    "animation": "dl-fadein 0.5s ease 1.5s forwards",
                    "opacity": "0",
                },
            ),
            rx.hstack(
                rx.spinner(size="1", color="#667eea"),
                rx.text(
                    "Распознаём элементы диаграммы...",
                    font_size="0.95em",
                    color="#2d3748",
                ),
                spacing="3",
                align="center",
                width="100%",
                style={
                    "animation": "dl-fadein 0.5s ease 5.0s forwards",
                    "opacity": "0",
                },
            ),
            spacing="3",
            align="start",
            width="100%",
            padding_x="0.5em",
        ),
        # Живой таймер обработки (heartbeat) — виден на долгом submit.
        rx.cond(
            State.progress_message,
            rx.hstack(
                rx.spinner(size="1", color="#667eea"),
                rx.text(
                    State.progress_message,
                    font_size="0.85em",
                    color="#4a5568",
                    font_weight="500",
                ),
                spacing="2",
                align="center",
                bg="#edf2ff",
                border="1px solid #c3dafe",
                border_radius="8px",
                padding_x="1em",
                padding_y="0.6em",
                width="100%",
            ),
        ),
        rx.cond(
            State.retry_message,
            rx.hstack(
                rx.spinner(size="1", color="#e07b00"),
                rx.text(
                    State.retry_message,
                    font_size="0.85em",
                    color="#e07b00",
                    font_weight="500",
                ),
                spacing="2",
                align="center",
                bg="#fffbeb",
                border="1px solid #fcd34d",
                border_radius="8px",
                padding_x="1em",
                padding_y="0.6em",
                width="100%",
            ),
        ),
        rx.button(
            rx.hstack(rx.icon("x", size=16), rx.text("Отменить"), spacing="2"),
            on_click=State.cancel_analysis,
            variant="outline",
            color_scheme="gray",
            size="2",
        ),
        spacing="5",
        align="center",
        width="100%",
        min_height="360px",
        justify="center",
    )


def index() -> rx.Component:
    return rx.box(
        navbar(),
        rx.container(
            rx.vstack(
                rx.vstack(
                    rx.heading(
                        "Анализ BPMN-диаграмм с помощью ИИ",
                        size="8",
                        text_align="center",
                        font_weight="bold",
                        color="white",
                        text_shadow="0 2px 4px rgba(0,0,0,0.2)",
                    ),
                    rx.text(
                        "Загрузите диаграмму — получите пошаговое описание всех элементов",
                        font_size="1.2em",
                        text_align="center",
                        color="#e2e8f0",
                    ),
                    spacing="3",
                    align="center",
                    width="100%",
                    padding_bottom="2em",
                ),
                rx.card(
                    rx.cond(
                        State.is_uploading,
                        loading_content(),
                        rx.vstack(
                            rx.hstack(
                                rx.icon("upload", size=30, color="#667eea"),
                                rx.heading(
                                    "Загрузка диаграммы", size="5", color="#1a202c"
                                ),
                                spacing="3",
                                align="center",
                            ),
                            rx.text(
                                "Поддерживаемый формат: PNG",
                                font_size="0.8em",
                                color="#718096",
                            ),
                            rx.divider(),
                            model_selector(),
                            rx.divider(),
                            rx.upload(
                                rx.vstack(
                                    rx.icon("image", size=40, color="#a0aec0"),
                                    rx.text(
                                        "Перетащите PNG-файл сюда или",
                                        font_size="0.9em",
                                        color="#4a5568",
                                    ),
                                    rx.button(
                                        "Выбрать файл",
                                        color_scheme="blue",
                                        size="3",
                                        variant="solid",
                                    ),
                                    rx.cond(
                                        State.uploaded_filename,
                                        rx.badge(
                                            f"Файл: {State.uploaded_filename}",
                                            color_scheme="green",
                                            variant="soft",
                                            font_size="0.8em",
                                        ),
                                    ),
                                    spacing="3",
                                    align="center",
                                ),
                                on_drop=State.handle_upload(
                                    rx.upload_files(upload_id="diagramlit_upload")
                                ),
                                id="diagramlit_upload",
                                multiple=False,
                                accept={"image/png": [".png"]},
                                border="2px dashed #cbd5e0",
                                border_radius="12px",
                                padding="2.5em",
                                bg="#fafafa",
                            ),
                            rx.cond(
                                State.uploaded_filename,
                                rx.button(
                                    rx.hstack(
                                        rx.icon("trash-2", size=16),
                                        rx.text("Очистить"),
                                    ),
                                    on_click=State.clear_upload,
                                    color_scheme="red",
                                    variant="outline",
                                    size="2",
                                ),
                            ),
                            spacing="5",
                            align="stretch",
                            width="100%",
                        ),
                    ),
                    padding="2em",
                    box_shadow="0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                    border_radius="16px",
                    bg="white",
                    min_height="380px",
                ),
                rx.grid(
                    rx.card(
                        rx.vstack(
                            rx.icon("brain", size=32, color="#667eea"),
                            rx.heading("ИИ-распознавание", size="4", color="#1a202c"),
                            rx.text(
                                "Автоматическое определение элементов BPMN-диаграммы",
                                text_align="center",
                                font_size="0.9em",
                                color="#4a5568",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="1.5em",
                        bg="white",
                        border_radius="12px",
                    ),
                    rx.card(
                        rx.vstack(
                            rx.icon("message-square", size=32, color="#667eea"),
                            rx.heading("Пошаговое описание", size="4", color="#1a202c"),
                            rx.text(
                                "Детальное объяснение всех процессов и связей",
                                text_align="center",
                                font_size="0.9em",
                                color="#4a5568",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="1.5em",
                        bg="white",
                        border_radius="12px",
                    ),
                    rx.card(
                        rx.vstack(
                            rx.icon("git-branch", size=32, color="#667eea"),
                            rx.heading("Формат BPMN", size="4", color="#1a202c"),
                            rx.text(
                                "Распознавание событий, задач, шлюзов и потоков",
                                text_align="center",
                                font_size="0.9em",
                                color="#4a5568",
                            ),
                            spacing="3",
                            align="center",
                        ),
                        padding="1.5em",
                        bg="white",
                        border_radius="12px",
                    ),
                    columns="repeat(3, 1fr)",
                    gap="6",
                    width="100%",
                    padding_y="3em",
                ),
                spacing="5",
                align="stretch",
                width="100%",
            ),
            max_width="1200px",
            margin="0 auto",
            padding_x="2em",
            padding_y="3em",
            min_height="calc(100vh - 140px)",
        ),
        footer(),
        width="100%",
        min_height="100vh",
    )


def bbox_overlay(box: BBox) -> rx.Component:
    """Красная рамка + прозрачный текст с белой обводкой и уменьшенным шрифтом."""
    return rx.box(
        rx.box(
            rx.text(
                box.label,
                font_size="0.65em",
                font_weight="900",
                color="#e53e3e",
                white_space="nowrap",
                text_shadow="1px 1px 0px rgba(255,255,255,0.9), -1px -1px 0px rgba(255,255,255,0.9), 1px -1px 0px rgba(255,255,255,0.9), -1px 1px 0px rgba(255,255,255,0.9)",
            ),
            position="absolute",
            top="-1.3em",
            left="-2px",
            bg="transparent",
            padding="0",
        ),
        position="absolute",
        left=f"{box.x * 100}%",
        top=f"{box.y * 100}%",
        width=f"{box.w * 100}%",
        height=f"{box.h * 100}%",
        border="2px solid #e53e3e",
        border_radius="3px",
        box_sizing="border-box",
        pointer_events="none",
    )


def diagram_viewer() -> rx.Component:
    """Окно с изображением и распознанными элементами (красные боксы)."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("scan-search", size=24, color="#667eea"),
                rx.heading("Распознанные элементы", size="4"),
                spacing="2",
                align="center",
            ),
            rx.text(
                "Красные рамки — обнаруженные элементы диаграммы",
                font_size="0.85em",
                color="#718096",
            ),
            rx.divider(),
            rx.box(
                rx.image(
                    src=State.image_data_url,
                    width="100%",
                    height="auto",
                    display="block",
                    border_radius="8px",
                ),
                rx.foreach(State.bounding_boxes, bbox_overlay),
                position="relative",
                width="100%",
                max_width="820px",
                margin="0 auto",
            ),
            spacing="3",
            width="100%",
            align="stretch",
        ),
        width="100%",
        padding="1.5em",
        border_radius="12px",
        bg="white",
    )


def llm_pending_banner() -> rx.Component:
    """Спиннер-таймер в блоке описания, пока LLM на устройстве ещё считает."""
    return rx.hstack(
        rx.spinner(size="1", color="#667eea"),
        rx.text(
            rx.cond(
                State.progress_message,
                State.progress_message,
                "ИИ готовит описание…",
            ),
            font_size="0.9em",
            color="#4a5568",
            font_weight="500",
        ),
        spacing="2",
        align="center",
        bg="#edf2ff",
        border="1px solid #c3dafe",
        border_radius="8px",
        padding_x="1em",
        padding_y="0.6em",
        width="100%",
    )


def steps_panel() -> rx.Component:
    """Окно с текстом всех шагов + кнопка копирования."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.icon("list-checks", size=24, color="#667eea"),
                    rx.heading("Пошаговое описание", size="4"),
                    spacing="2",
                    align="center",
                ),
                rx.spacer(),
                rx.button(
                    rx.hstack(
                        rx.icon("copy", size=16),
                        rx.text("Копировать"),
                        spacing="2",
                        align="center",
                    ),
                    on_click=[
                        rx.set_clipboard(State.steps_text),
                        rx.toast.success("Текст скопирован в буфер обмена"),
                    ],
                    color_scheme="blue",
                    variant="soft",
                    size="2",
                ),
                width="100%",
                align="center",
            ),
            rx.divider(),
            # Пока описание готовится на устройстве — спиннер-таймер сверху.
            rx.cond(
                State.llm_status == "pending",
                llm_pending_banner(),
            ),
            rx.box(
                rx.vstack(
                    rx.foreach(
                        State.step_by_step_description,
                        lambda step, idx: rx.hstack(
                            rx.center(
                                rx.text(
                                    idx + 1,
                                    font_size="0.8em",
                                    font_weight="700",
                                    color="white",
                                ),
                                min_width="1.6em",
                                height="1.6em",
                                bg="#667eea",
                                border_radius="50%",
                                flex_shrink="0",
                            ),
                            rx.text(
                                step,
                                font_size="0.95em",
                                color="#2d3748",
                                line_height="1.5",
                            ),
                            spacing="3",
                            align="start",
                            width="100%",
                        ),
                    ),
                    spacing="3",
                    align="start",
                    width="100%",
                ),
                width="100%",
                max_height="420px",
                overflow_y="auto",
                padding="1.25em",
                bg="#f7fafc",
                border="1px solid #e2e8f0",
                border_radius="10px",
            ),
            spacing="3",
            width="100%",
            align="stretch",
        ),
        width="100%",
        padding="1.5em",
        border_radius="12px",
        bg="white",
    )


def security_panel() -> rx.Component:
    """Панель с уязвимостями архитектуры, видимыми на диаграмме."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("shield-alert", size=24, color="#e53e3e"),
                rx.heading("Проблемы безопасности", size="4"),
                spacing="2",
                align="center",
            ),
            rx.divider(),
            rx.cond(
                State.security_issues,
                rx.box(
                    rx.vstack(
                        rx.foreach(
                            State.security_issues,
                            lambda issue: rx.hstack(
                                rx.icon(
                                    "triangle-alert",
                                    size=16,
                                    color="#e53e3e",
                                    flex_shrink="0",
                                    margin_top="2px",
                                ),
                                rx.text(
                                    issue,
                                    font_size="0.9em",
                                    color="#2d3748",
                                    line_height="1.5",
                                ),
                                spacing="3",
                                align="start",
                                width="100%",
                            ),
                        ),
                        spacing="3",
                        align="start",
                        width="100%",
                    ),
                    width="100%",
                    max_height="420px",
                    overflow_y="auto",
                    padding="1.25em",
                    bg="#fff5f5",
                    border="1px solid #fed7d7",
                    border_radius="10px",
                ),
                rx.center(
                    rx.vstack(
                        rx.icon("shield-check", size=40, color="#38a169"),
                        rx.text(
                            "Уязвимостей не обнаружено",
                            font_size="0.95em",
                            font_weight="600",
                            color="#38a169",
                        ),
                        spacing="3",
                        align="center",
                    ),
                    padding="2em",
                    bg="#f0fff4",
                    border="1px solid #c6f6d5",
                    border_radius="10px",
                    width="100%",
                ),
            ),
            spacing="3",
            width="100%",
            align="stretch",
        ),
        width="100%",
        padding="1.5em",
        border_radius="12px",
        bg="white",
    )


def no_results_placeholder() -> rx.Component:
    """Вместо вечного спиннера: понятное состояние «результатов нет»."""
    return rx.center(
        rx.vstack(
            rx.icon("file-question", size=48, color="white"),
            rx.text(
                "Результатов анализа пока нет",
                color="white",
                font_size="1.2em",
                font_weight="600",
            ),
            rx.text(
                "Загрузите диаграмму на главной странице",
                color="#e2e8f0",
                font_size="0.95em",
            ),
            rx.link(
                rx.button(
                    rx.hstack(rx.icon("arrow-left", size=16), rx.text("На главную")),
                    variant="solid",
                    color_scheme="blue",
                    size="3",
                ),
                href="/",
            ),
            spacing="4",
            align="center",
        ),
        min_height="60vh",
    )


def results() -> rx.Component:
    return rx.box(
        navbar(),
        rx.container(
            rx.cond(
                State.diagram_type,
                rx.vstack(
                    rx.hstack(
                        rx.link(
                            rx.button(
                                rx.hstack(
                                    rx.icon("arrow-left", size=16), rx.text("Назад")
                                ),
                                variant="outline",
                                bg="white",
                            ),
                            href="/",
                        ),
                        rx.heading("Результаты анализа", size="7", color="white"),
                        spacing="4",
                        align="center",
                        width="100%",
                    ),
                    rx.divider(),
                    rx.cond(
                        State.has_error,
                        rx.card(
                            rx.hstack(
                                rx.icon("circle-x", size=28, color="#e53e3e"),
                                rx.vstack(
                                    rx.heading(
                                        "Ошибка при обработке",
                                        size="4",
                                        color="#e53e3e",
                                    ),
                                    rx.text(
                                        State.error_message,
                                        font_size="0.9em",
                                        color="#2d3748",
                                        word_break="break-word",
                                    ),
                                    spacing="1",
                                    align="start",
                                ),
                                spacing="4",
                                align="start",
                                width="100%",
                            ),
                            width="100%",
                            padding="1.5em",
                            border_radius="12px",
                            bg="#fff5f5",
                            border="1px solid #fed7d7",
                        ),
                    ),
                    rx.cond(
                        ~State.has_error,
                        rx.vstack(
                            rx.card(
                                rx.vstack(
                                    rx.hstack(
                                        rx.icon(
                                            "bar-chart-2", size=24, color="#667eea"
                                        ),
                                        rx.heading("Тип диаграммы", size="4"),
                                        spacing="2",
                                        align="center",
                                    ),
                                    rx.hstack(
                                        rx.badge(
                                            State.diagram_type,
                                            color_scheme="blue",
                                            variant="soft",
                                            size="3",
                                        ),
                                        rx.text(
                                            f"Модель: {State.model_label}",
                                            font_size="0.8em",
                                            color="#718096",
                                        ),
                                        spacing="3",
                                        align="center",
                                    ),
                                ),
                                spacing="3",
                                width="100%",
                                padding="1.5em",
                                border_radius="12px",
                                bg="white",
                            ),
                            rx.cond(State.image_data_url, diagram_viewer()),
                            rx.cond(
                                State.detected_elements,
                                rx.card(
                                    rx.vstack(
                                        rx.hstack(
                                            rx.icon(
                                                "layout-grid", size=24, color="#667eea"
                                            ),
                                            rx.heading("Список элементов", size="4"),
                                            spacing="2",
                                            align="center",
                                        ),
                                        rx.flex(
                                            rx.foreach(
                                                State.detected_elements,
                                                lambda el: rx.badge(
                                                    el,
                                                    color_scheme="green",
                                                    variant="soft",
                                                    size="2",
                                                    padding_x="1em",
                                                    padding_y="0.5em",
                                                    border_radius="8px",
                                                ),
                                            ),
                                            wrap="wrap",
                                            spacing="2",
                                            gap="2",
                                        ),
                                    ),
                                    spacing="3",
                                    width="100%",
                                    padding="1.5em",
                                    border_radius="12px",
                                    bg="white",
                                ),
                            ),
                            rx.cond(
                                State.relationships,
                                rx.card(
                                    rx.vstack(
                                        rx.hstack(
                                            rx.icon(
                                                "share-2", size=24, color="#667eea"
                                            ),
                                            rx.heading(
                                                "Связи между элементами", size="4"
                                            ),
                                            spacing="2",
                                            align="center",
                                        ),
                                        rx.vstack(
                                            rx.foreach(
                                                State.relationships,
                                                lambda rel: rx.hstack(
                                                    rx.icon(
                                                        "link-2",
                                                        size=14,
                                                        color="#764ba2",
                                                    ),
                                                    rx.text(
                                                        rel,
                                                        font_size="0.95em",
                                                        color="#2d3748",
                                                    ),
                                                    spacing="2",
                                                    align="center",
                                                ),
                                            ),
                                            spacing="2",
                                            align="start",
                                            width="100%",
                                        ),
                                    ),
                                    spacing="3",
                                    width="100%",
                                    padding="1.5em",
                                    border_radius="12px",
                                    bg="white",
                                ),
                            ),
                            # Блок описания: показываем, если есть шаги ИЛИ ещё идёт генерация
                            rx.cond(
                                State.step_by_step_description
                                | (State.llm_status == "pending"),
                                rx.grid(
                                    steps_panel(),
                                    security_panel(),
                                    columns="2",
                                    gap="6",
                                    width="100%",
                                ),
                            ),
                            spacing="6",
                            align="stretch",
                            width="100%",
                        ),
                    ),
                    spacing="4",
                    align="stretch",
                    width="100%",
                ),
                rx.cond(
                    State.is_uploading,
                    rx.center(
                        rx.vstack(
                            rx.hstack(
                                rx.spinner(size="3", color="#667eea"),
                                rx.text(
                                    "Анализ ещё выполняется...",
                                    color="white",
                                    font_size="1.1em",
                                ),
                                spacing="3",
                                align="center",
                            ),
                            rx.cond(
                                State.progress_message,
                                rx.text(
                                    State.progress_message,
                                    color="#e2e8f0",
                                    font_size="0.9em",
                                ),
                            ),
                            rx.button(
                                rx.hstack(rx.icon("x", size=16), rx.text("Отменить")),
                                on_click=State.cancel_analysis,
                                variant="outline",
                                bg="white",
                                size="2",
                            ),
                            spacing="4",
                            align="center",
                        ),
                        min_height="60vh",
                    ),
                    no_results_placeholder(),
                ),
            ),
            max_width="1400px",
            margin="0 auto",
            padding_x="2em",
            padding_y="3em",
            min_height="calc(100vh - 140px)",
        ),
        footer(),
        width="100%",
        min_height="100vh",
    )


style = {
    "font_family": "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "background_attachment": "fixed",
}

app = rx.App(style=style)
app.add_page(index, route="/", title="DiagramLIT — Анализ диаграмм")
app.add_page(results, route="/results", title="Результаты — DiagramLIT")
