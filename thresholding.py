"""
thresholding.py — алгоритмы пороговой обработки (чистая математика, без интерфейса).

Вход везде: gray — 2D-массив numpy uint8 (яркость 0..255).
Правило бинаризации: пиксель > T -> 255 (белый), иначе 0 (чёрный).
"""
import numpy as np


def histogram(gray: np.ndarray) -> np.ndarray:
    """Гистограмма яркости: hist[i] = число пикселей с яркостью i (256 ячеек)."""
    return np.bincount(gray.ravel(), minlength=256)


def otsu(hist: np.ndarray) -> int:
    """
    Глобальный метод 1: Оцу.
    Для каждого кандидата t делим гистограмму на классы (<=t и >t) и считаем
    межклассовую дисперсию  sigma^2(t) = w1 * w2 * (m1 - m2)^2,
    где w — число пикселей класса, m — его средняя яркость. Берём t с максимумом.
    """
    hist = hist.astype(np.float64)
    levels = np.arange(256)
    w1 = np.cumsum(hist)                    # размер класса 1 для каждого t
    w2 = hist.sum() - w1                    # размер класса 2
    s1 = np.cumsum(hist * levels)           # сумма яркостей класса 1
    s2 = s1[-1] - s1                        # сумма яркостей класса 2
    valid = (w1 > 0) & (w2 > 0)             # пустой класс считать нельзя (деление на 0)
    m1 = np.divide(s1, w1, out=np.zeros(256), where=valid)
    m2 = np.divide(s2, w2, out=np.zeros(256), where=valid)
    var = np.where(valid, w1 * w2 * (m1 - m2) ** 2, -1.0)
    return int(np.argmax(var))


def iterative(hist: np.ndarray, eps: float = 0.5) -> int:
    """
    Глобальный метод 2: итеративный.
    1) T = средняя яркость; 2) делим пиксели по T на два класса, берём их средние
    m1 и m2; 3) T = (m1 + m2) / 2; повторяем, пока T меняется больше чем на eps.
    """
    hist = hist.astype(np.float64)
    levels = np.arange(256)
    T = (hist * levels).sum() / hist.sum()
    for _ in range(100):                    # защита от зацикливания
        low = levels <= T
        n1, n2 = hist[low].sum(), hist[~low].sum()
        if n1 == 0 or n2 == 0:              # однотонное изображение
            break
        m1 = (hist[low] * levels[low]).sum() / n1
        m2 = (hist[~low] * levels[~low]).sum() / n2
        new_T = (m1 + m2) / 2
        done = abs(new_T - T) < eps
        T = new_T
        if done:
            break
    return int(round(T))


def apply_global(gray: np.ndarray, T: float, invert: bool = False) -> np.ndarray:
    """Бинаризация одним порогом T. invert меняет чёрное и белое местами."""
    mask = gray > T
    if invert:
        mask = ~mask
    return mask.astype(np.uint8) * 255


def _axis_weights(n_pix: int, block: int, n_grid: int):
    """
    Для каждой координаты пикселя: индексы двух ближайших узлов сетки порогов
    и вес для линейной интерполяции. Узел сетки = центр блока.
    """
    g = np.clip((np.arange(n_pix) + 0.5) / block - 0.5, 0, n_grid - 1)
    i0 = np.floor(g).astype(int)
    i1 = np.minimum(i0 + 1, n_grid - 1)
    return i0, i1, g - i0


def adaptive(gray: np.ndarray, block: int, min_std: float, invert: bool = False):
    """
    Адаптивная пороговая обработка (разбиение на подизображения).
    1) Делим изображение на блоки block x block.
    2) В блоке с достаточным разбросом (СКО >= min_std) порог = Оцу по гистограмме блока.
       Блок с малым СКО — только фон или только объект: гистограмма унимодальна,
       порог по ней был бы случайным.
    3) Такие блоки получают среднее по известным соседям (волной от «хороших» блоков).
    4) Пороги привязаны к центрам блоков; для каждого пикселя порог интерполируется
       билинейно, поэтому на границах блоков нет «швов».
    Возвращает (бинарное изображение, карта порогов uint8).
    """
    h, w = gray.shape
    gh, gw = -(-h // block), -(-w // block)         # деление с округлением вверх
    grid = np.full((gh, gw), np.nan)                # NaN = порог ещё не известен

    # --- шаги 1–2: порог в каждом информативном блоке ---
    for by in range(gh):
        for bx in range(gw):
            tile = gray[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            if tile.std() >= min_std:
                grid[by, bx] = otsu(histogram(tile))

    # --- шаг 3: заполнение блоков без порога ---
    if np.isnan(grid).all():                        # запасной вариант: глобальный Оцу
        grid[:] = otsu(histogram(gray))
    while np.isnan(grid).any():
        p = np.pad(grid, 1, constant_values=np.nan)
        # четыре соседа каждого блока (сверху, снизу, слева, справа)
        nb = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        cnt = (~np.isnan(nb)).sum(axis=0)
        total = np.nansum(nb, axis=0)
        fill = np.isnan(grid) & (cnt > 0)
        grid[fill] = total[fill] / cnt[fill]

    # --- шаг 4: билинейная интерполяция порога по всему изображению ---
    y0, y1, yt = _axis_weights(h, block, gh)
    x0, x1, xt = _axis_weights(w, block, gw)
    top = grid[y0][:, x0] * (1 - xt) + grid[y0][:, x1] * xt     # интерполяция по x
    bot = grid[y1][:, x0] * (1 - xt) + grid[y1][:, x1] * xt
    T_map = top * (1 - yt)[:, None] + bot * yt[:, None]          # затем по y

    mask = gray > T_map
    if invert:
        mask = ~mask
    return mask.astype(np.uint8) * 255, np.clip(T_map, 0, 255).astype(np.uint8)


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """
    Среднее в окне (2r+1) x (2r+1) вокруг каждого пикселя за O(1) на пиксель.
    Используется интегральное изображение: c[y, x] = сумма всех пикселей выше и левее,
    тогда сумма в любом прямоугольнике = c[D] - c[B] - c[C] + c[A].
    Края: значения за границей берутся как ближайшие пиксели (mode="edge").
    """
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), r, mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))          # интегральное изображение
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def smooth(gray: np.ndarray, r: int) -> np.ndarray:
    """Сглаживание усредняющим фильтром радиуса r (r = 0 — без изменений). Подавляет шум."""
    if r <= 0:
        return gray
    return np.clip(np.rint(box_mean(gray, r)), 0, 255).astype(np.uint8)


def adaptive_local(gray: np.ndarray, radius: int, a: float, b: float, invert: bool = False):
    """
    Адаптивная обработка по локальным статистикам (Гонсалес и Вудс, «переменный порог»).
    В окне вокруг пикселя считаются среднее m и СКО s; пиксель белый, если
        v > a * s   И   v > b * m.
    Условие v > b*m сравнивает пиксель с локальным фоном, поэтому тень или градиент
    освещения не мешают; v > a*s отсекает пиксели в областях с сильным разбросом.
    Возвращает (бинарное изображение, карта порогов b*m).
    """
    g = gray.astype(np.float64)
    m = box_mean(g, radius)
    s = np.sqrt(np.maximum(box_mean(g * g, radius) - m * m, 0))   # Var = E[v^2] − (E[v])^2
    mask = (g > a * s) & (g > b * m)
    if invert:
        mask = ~mask
    return mask.astype(np.uint8) * 255, np.clip(b * m, 0, 255).astype(np.uint8)
