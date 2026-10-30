"""
histogram.py — модуль визуализации гистограмм (Presentation Layer).

Архитектура: согласно методичке, это часть Presentation Layer (Слой представления).
Изолирует математику алгоритмов (thresholding.py, pointwise.py) от UI.
"""

import matplotlib
# КРИТИЧНО: бэкенд "Agg" рендерит в память, а не открывает GUI-окно.
# Streamlit — веб-сервер, у него нет монитора. Без этой строки приложение зависнет.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(hist, lines):
    """
    Гистограмма яркости с вертикальными линиями порогов.
    Используется в разделе "Пороговая обработка" для сравнения методов Оцу и Итеративного.
    Показывает, насколько близки пороги, найденные разными алгоритмами.
    Параметры:
        hist — массив из 256 значений (количество пикселей каждой яркости)
        lines — список (порог, цвет, подпись) для вертикальных линий
    """
    fig, ax = plt.subplots(figsize=(4, 2.2))

    # Столбчатая диаграмма распределения яркостей
    ax.bar(range(256), hist, width=1.0, color="#6b7780")

    # Вертикальные линии показывают найденные пороги
    for value, color, label in lines:
        ax.axvline(value, color=color, linewidth=2, label=f"{label}: {value}")

    # Жесткая фиксация диапазона 0-255 для корректного сравнения
    # Без этого matplotlib автоподберет диапазон под данные, исказив восприятие
    ax.set_xlim(0, 255)
    ax.set_xlabel("Яркость")
    ax.set_ylabel("Пикселей")
    ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


def plot_pair(h_before, h_after, title_before="До", title_after="После"):
    """
    Две гистограммы рядом для сравнения "До/После" обработки.
    Используется в разделе "Поэлементные операции и контраст".
    Наглядно показывает эффект операций (сдвиг, растяжение гистограммы).
    Параметры:
        h_before, h_after — гистограммы до и после обработки
    """
    # sharex=True КРИТИЧЕН: оба графика используют одну ось X (0-255)
    # Без этого matplotlib масштабирует оси отдельно, сравнение станет некорректным
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2), sharex=True)

    for ax, h, t in zip(axes, (h_before, h_after), (title_before, title_after)):
        ax.bar(range(256), h, width=1.0, color="#6b7780")
        ax.set_title(t, fontsize=9)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Яркость")

    fig.tight_layout()
    return fig