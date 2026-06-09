import reflex as rx
from pydantic import BaseModel
import asyncio
import base64
from typing import List, Dict, Any


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
    Координаты — ДОЛИ от размеров изображения (0..1), а не пиксели.
    Если модель вернула пиксели — поделите на ширину/высоту изображения."""

    label: str
    x: float
    y: float
    w: float
    h: float


def _demo_result(diagram_type: str = "BPMN Process Diagram") -> Dict[str, Any]:
    """Временная заглушка. Замени реальным выводом модели (см. _infer_*)."""
    return {
        "diagram_type": diagram_type,
        "detected_elements": [
            "Start Event",
            "Task: Проверка заказа",
            "Gateway: Товар в наличии?",
            "Task: Оплата",
            "End Event",
        ],
        "relationships": [
            "Start → Проверка заказа",
            "Проверка заказа → Gateway",
            "Gateway → Оплата",
            "Оплата → End",
        ],
        "bounding_boxes": [
            {"label": "Start", "x": 0.04, "y": 0.40, "w": 0.09, "h": 0.16},
            {"label": "Проверка заказа", "x": 0.22, "y": 0.33, "w": 0.20, "h": 0.30},
            {"label": "Gateway", "x": 0.50, "y": 0.36, "w": 0.11, "h": 0.24},
            {"label": "Оплата", "x": 0.68, "y": 0.33, "w": 0.20, "h": 0.30},
            {"label": "End", "x": 0.91, "y": 0.41, "w": 0.07, "h": 0.14},
        ],
        "step_by_step_description": [
            "Процесс начинается со стартового события — поступает новый заказ.",
            "Задача «Проверка заказа»: система проверяет корректность данных.",
            "Шлюз (Gateway): проверяется наличие товара на складе.",
            "Если товар есть — выполняется задача «Оплата».",
            "Конечное событие: процесс завершается, заказ переходит в «Оплачен».",
        ],
    }


class State(rx.State):
    """Состояние приложения DiagramLIT"""

    is_uploading: bool = False
    is_processing: bool = False

    selected_model: str = "gemini"
    image_data_url: str = ""

    uploaded_filename: str = ""
    diagram_type: str = ""
    detected_elements: List[str] = []
    relationships: List[str] = []
    step_by_step_description: List[str] = []
    bounding_boxes: List[BBox] = []

    # ── Вспомогательное ────────────────────────────────────────────────
    @rx.var
    def model_label(self) -> str:
        return MODEL_BACKENDS.get(self.selected_model, self.selected_model)

    @rx.var
    def steps_text(self) -> str:
        """Все шаги одним текстом — для отображения и копирования."""
        return "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(self.step_by_step_description)
        )

    def set_selected_model(self, value: str):
        self.selected_model = value

    # ── Точки интеграции реальных моделей ───────────────────────────────
    # Каждая функция принимает байты картинки и возвращает словарь с полями:
    #   diagram_type, detected_elements, relationships,
    #   bounding_boxes (список {label, x, y, w, h} в долях 0..1),
    #   step_by_step_description

    async def _infer_gemini(self, image_bytes: bytes) -> Dict[str, Any]:
        """Облачный инференс через Gemini 3.1 Pro.
        Зависимости: pip install google-genai; ключ в GEMINI_API_KEY.
        В промпте попроси нормализованные координаты боксов."""
        # import os
        # from google import genai
        # client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # response = client.models.generate_content(
        #     model="gemini-3.1-pro",
        #     contents=[{
        #         "role": "user",
        #         "parts": [
        #             {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
        #             {"text": PROMPT_RU},
        #         ],
        #     }],
        # )
        # return _parse_model_json(response.text)
        await asyncio.sleep(1.0)
        return _demo_result()

    async def _infer_local_vlm(self, image_bytes: bytes) -> Dict[str, Any]:
        """Локальная VLM на GPU с 8 ГБ VRAM.
        Рекомендация: Qwen2.5-VL-7B-Instruct в 4-bit — хорошо понимает русский
        и помещается в ~6-7 ГБ. Pixtral-12B в 8 ГБ влезает плохо.
        ВАЖНО: инференс блокирующий — выноси в asyncio.to_thread."""
        # result = await asyncio.to_thread(self._run_local_vlm_blocking, image_bytes)
        # return result
        await asyncio.sleep(1.5)
        return _demo_result()

    async def _infer_orangepi(self, image_bytes: bytes) -> Dict[str, Any]:
        """Пайплайн для OrangePi RV2 (RISC-V, без CUDA):
          1. YOLOv8 (onnx/ncnn) — детекция блоков (тут боксы);
          2. OCR (PaddleOCR / Tesseract rus) — текст внутри блоков;
          3. лёгкая LLM (Qwen2.5-0.5B/1.5B в GGUF) — оборачивает JSON в текст.
        YOLO даёт пиксели — дели на ширину/высоту для нормализации."""
        # result = await asyncio.to_thread(self._run_orangepi_blocking, image_bytes)
        # return result
        await asyncio.sleep(2.0)
        return _demo_result()

    async def handle_upload(self, files: List[rx.UploadFile]):
        if not files:
            return

        self.is_uploading = True
        self.uploaded_filename = files[0].filename
        yield

        try:
            file_content = await files[0].read()

            self.image_data_url = (
                "data:image/png;base64," + base64.b64encode(file_content).decode()
            )

            if self.selected_model == "gemini":
                result = await self._infer_gemini(file_content)
            elif self.selected_model == "local_vlm":
                result = await self._infer_local_vlm(file_content)
            elif self.selected_model == "orangepi":
                result = await self._infer_orangepi(file_content)
            else:
                result = _demo_result("Неизвестная модель")

            self.diagram_type = result["diagram_type"]
            self.detected_elements = result["detected_elements"]
            self.relationships = result["relationships"]
            self.step_by_step_description = result["step_by_step_description"]
            self.bounding_boxes = [BBox(**b) for b in result.get("bounding_boxes", [])]

        except Exception as e:
            print(f"Ошибка: {e}")
            self.diagram_type = "Ошибка"
        finally:
            self.is_uploading = False

        yield rx.redirect("/results")

    def clear_upload(self):
        self.uploaded_filename = ""
        self.image_data_url = ""
        self.diagram_type = ""
        self.detected_elements = []
        self.relationships = []
        self.step_by_step_description = []
        self.bounding_boxes = []


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


def index() -> rx.Component:
    return rx.fragment(
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
                    rx.vstack(
                        rx.hstack(
                            rx.icon("upload", size=30, color="#667eea"),
                            rx.heading("Загрузка диаграммы", size="5", color="#1a202c"),
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
                            on_drop=State.handle_upload,
                            multiple=False,
                            accept={"image/png": [".png"]},
                            border="2px dashed #cbd5e0",
                            border_radius="12px",
                            padding="2.5em",
                            bg="#fafafa",
                        ),
                        rx.cond(
                            State.is_uploading,
                            rx.center(
                                rx.spinner(size="3", color="#667eea"),
                                rx.text(
                                    f"Обработка через {State.model_label}...",
                                    margin_left="1em",
                                    color="#4a5568",
                                ),
                                padding="2em",
                            ),
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
                    ),
                    spacing="5",
                    align="stretch",
                    width="100%",
                    padding="2em",
                    box_shadow="0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                    border_radius="16px",
                    bg="white",
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
            ),
            max_width="1200px",
            padding_x="2em",
            padding_y="3em",
            min_height="calc(100vh - 140px)",
        ),
        footer(),
    )


def bbox_overlay(box: BBox) -> rx.Component:
    """Одна красная рамка поверх изображения + подпись."""
    return rx.box(
        rx.box(
            rx.text(
                box.label,
                font_size="0.7em",
                font_weight="600",
                color="white",
                white_space="nowrap",
            ),
            position="absolute",
            top="-1.45em",
            left="-2px",
            bg="#e53e3e",
            padding_x="0.4em",
            padding_y="0.05em",
            border_radius="4px 4px 0 0",
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


def results() -> rx.Component:
    return rx.fragment(
        navbar(),
        rx.container(
            rx.cond(
                State.diagram_type,
                rx.vstack(
                    rx.hstack(
                        rx.link(
                            rx.button(
                                rx.hstack(
                                    rx.icon("arrow-left", size=16),
                                    rx.text("Назад"),
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
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.icon("bar-chart-2", size=24, color="#667eea"),
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
                                    rx.icon("grid", size=24, color="#667eea"),
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
                                    rx.icon("share-2", size=24, color="#667eea"),
                                    rx.heading("Связи между элементами", size="4"),
                                    spacing="2",
                                    align="center",
                                ),
                                rx.vstack(
                                    rx.foreach(
                                        State.relationships,
                                        lambda rel: rx.hstack(
                                            rx.icon("link-2", size=14, color="#764ba2"),
                                            rx.text(rel, font_size="0.95em", color="#2d3748"),
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
                    rx.cond(State.step_by_step_description, steps_panel()),
                    spacing="6",
                    align="stretch",
                ),
                rx.center(
                    rx.hstack(
                        rx.spinner(size="3", color="#667eea"),
                        rx.text("Загрузка результатов...", color="white", font_size="1.1em"),
                        spacing="3",
                        align="center",
                    ),
                    min_height="60vh",
                ),
            ),
            max_width="1000px",
            padding_x="2em",
            padding_y="3em",
            min_height="calc(100vh - 140px)",
        ),
        footer(),
    )

style = {
    "font_family": "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "background_attachment": "fixed",
}

app = rx.App(style=style)
app.add_page(index, route="/", title="DiagramLIT — Анализ диаграмм")
app.add_page(results, route="/results", title="Результаты — DiagramLIT")
