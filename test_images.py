"""
test_images.py — «база» тестовых изображений, генерируемых программно.

IMAGES   — изображения для пороговой обработки. make(имя) возвращает (изображение, эталон):
           эталон — булев массив «пиксель должен быть белым». По нему приложение считает
           точность каждого метода (доля правильно классифицированных пикселей).
CONTRAST — изображения для линейного контрастирования и поэлементных операций.
Генератор шума инициализирован фиксированным числом, поэтому результат повторяется.
Запуск `python test_images.py` сохраняет все изображения в папку images/ как PNG-файлы.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 480, 320
_YY, _XX = np.mgrid[:H, :W]


def _font(size):
    """Шрифт с кириллицей, если найден; иначе встроенный шрифт Pillow."""
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _text_alpha(blur=0.0):
    """Маска текста 0..1. Возвращает (маска, маска_без_размытия) — вторая нужна для эталона."""
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.text((30, 70), "Пороговая", fill=255, font=_font(54))
    d.text((30, 150), "обработка", fill=255, font=_font(54))
    d.text((30, 250), "Lab 3, вариант 6", fill=255, font=_font(28))
    sharp = np.asarray(m) / 255.0
    if blur:
        m = m.filter(ImageFilter.GaussianBlur(blur))      # имитация расфокусировки
    return np.asarray(m) / 255.0, sharp


def _render(bg, fg, blur=0.0):
    """Текст яркости fg на фоне bg. Эталон: белое = фон (текст должен стать чёрным)."""
    a, sharp = _text_alpha(blur)
    return bg * (1 - a) + fg * a, sharp < 0.5


def _noise(arr, sigma, rng):
    """Гауссов шум со средним 0 и СКО sigma."""
    return arr + rng.normal(0, sigma, arr.shape)


def _u8(arr):
    """Обрезка до 0..255 и приведение к uint8."""
    return np.clip(arr, 0, 255).astype(np.uint8)


# ----------------------------- пороговая обработка -----------------------------
def _clean(rng):
    arr, t = _render(210, 40)
    return _u8(arr), t


def _uneven(rng):
    # Освещённость падает слева направо со 100% до 22%: тёмный фон справа темнее
    # светлого текста слева -> один глобальный порог невозможен.
    arr, t = _render(220, 40)
    return _u8(arr * (1 - 0.78 * np.linspace(0, 1, W))[None, :]), t


def _noisy(rng):
    arr, t = _render(200, 60)
    arr = arr * (1 - 0.5 * np.linspace(0, 1, H))[:, None]      # градиент сверху вниз
    return _u8(_noise(arr, 25, rng)), t


def _low_contrast(rng):
    arr, t = _render(130, 105)
    return _u8(_noise(arr, 4, rng)), t


def _blurred(rng):
    arr, t = _render(210, 40, blur=3.5)
    return _u8(_noise(arr, 5, rng)), t


def _scene(strip_left, rng):
    """
    Три уровня яркости: тёмная полоса (43, ~8 % площади), основной фон/предмет (154)
    и яркий диск (233, ~13 %). strip_left: полоса слева (иначе снизу).
    Возвращает яркость и маски полосы и диска.
    """
    arr = np.full((H, W), 154.0)
    strip = (_XX < int(0.08 * W)) if strip_left else (_YY > int(0.92 * H))
    disc = (_XX - (300 if strip_left else 150)) ** 2 + (_YY - 160) ** 2 < 80 ** 2
    arr[strip] = 43
    arr[disc] = 233
    return _u8(_noise(arr, 15, rng)), strip, disc


def _otsu_better(rng):
    # Предмет (средняя яркость + блик) на узкой тёмной полосе фона: эталон = всё, кроме полосы.
    img, strip, disc = _scene(True, rng)
    return img, ~strip


def _iter_better(rng):
    # Яркий диск на фоне с тенью: эталон = только диск.
    img, strip, disc = _scene(False, rng)
    return img, disc


IMAGES = {
    "1. Чистый текст (все методы справляются)": _clean,
    "2. Неравномерное освещение (только адаптивный)": _uneven,
    "3. Шум + градиент освещённости": _noisy,
    "4. Малый контраст": _low_contrast,
    "5. Размытый текст": _blurred,
    "6. Оцу лучше итеративного (предмет с бликом)": _otsu_better,
    "7. Итеративный лучше Оцу (яркий диск и тень)": _iter_better,
}


def make(name):
    """Возвращает (изображение uint8, эталонная маска) по названию."""
    return IMAGES[name](np.random.default_rng(42))


# ----------------------------- контрастирование -----------------------------
def _pattern():
    """Исходный «сюжет» 0..1: плавная волна и три круга разной яркости."""
    p = 0.5 + 0.25 * np.sin(_XX / 35.0) * np.cos(_YY / 28.0)
    for cx, cy, r, v in [(120, 110, 60, 0.9), (330, 200, 70, 0.1), (380, 90, 40, 0.75)]:
        p[(_XX - cx) ** 2 + (_YY - cy) ** 2 < r * r] = v
    return p


def _range_image(lo, hi, rng, outliers=False):
    """Тот же сюжет, сжатый в диапазон яркостей [lo, hi]."""
    arr = _noise(lo + (hi - lo) * _pattern(), 1.5, rng)
    img = _u8(arr)
    if outliers:                                   # несколько «битых» пикселей 0 и 255
        img[5, 5:10] = 0
        img[8, 5:10] = 255
    return img


CONTRAST = {
    "1. Малоконтрастное (яркости 100–150)": lambda r: _range_image(100, 150, r),
    "2. Пересвеченное (яркости 180–255)": lambda r: _range_image(180, 255, r),
    "3. Тёмное (яркости 0–70)": lambda r: _range_image(0, 70, r),
    "4. Малый контраст + выбросы 0 и 255": lambda r: _range_image(100, 150, r, outliers=True),
}


def make_contrast(name):
    """Возвращает изображение uint8 для контрастирования по названию."""
    return CONTRAST[name](np.random.default_rng(42))


def save_all(folder="images"):
    """Сохраняет всю базу в PNG-файлы (для отчёта и сдачи)."""
    os.makedirs(folder, exist_ok=True)
    for name in IMAGES:
        Image.fromarray(make(name)[0]).save(os.path.join(folder, "thr_" + name[0] + ".png"))
    for name in CONTRAST:
        Image.fromarray(make_contrast(name)).save(os.path.join(folder, "contrast_" + name[0] + ".png"))


if __name__ == "__main__":
    save_all()
