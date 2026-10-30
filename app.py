"""
app.py — веб-интерфейс на Streamlit (слой представления).
Запуск:  streamlit run app.py
Здесь только ввод параметров и вывод картинок. Вся математика лежит в
thresholding.py (пороговая обработка) и pointwise.py (поэлементные операции и контраст).
Streamlit при любом действии пользователя выполняет этот файл заново сверху вниз.
"""
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import histogram
import pointwise
import test_images
import thresholding as th

MAX_SIDE = 800    # большие фото уменьшаем, чтобы страница не тормозила

st.set_page_config(page_title="Обработка изображений", layout="wide")


# ---------------------------- общие вспомогательные функции ----------------------------
def to_gray(img: Image.Image) -> np.ndarray:
    """Загруженное изображение -> градации серого (L = 0.299R + 0.587G + 0.114B), с уменьшением."""
    img = img.convert("L")
    k = min(1.0, MAX_SIDE / max(img.size))
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)))
    return np.asarray(img)


def timed(fn, *args):
    """Вызывает fn и возвращает (результат, время в мс)."""
    t0 = time.perf_counter()
    res = fn(*args)
    return res, (time.perf_counter() - t0) * 1000


def stats(g):
    """Краткая статистика яркости для подписей."""
    return f"min {g.min()}, max {g.max()}, среднее {g.mean():.0f}, СКО {g.std():.1f}"


# ---------------------------- страница 1: пороговая обработка ----------------------------
def page_threshold(upload):
    st.header("Пороговая обработка")
    st.caption("Глобальные методы (Оцу, итеративный) и адаптивные — на одном изображении.")
    with st.sidebar:
        name = st.selectbox("Тестовое изображение", list(test_images.IMAGES)) # создает выпадающий список.
        smooth_r = st.slider("Сглаживание перед порогом (радиус, 0 — выкл.)", 0, 10, 0)
        invert = st.checkbox("Инвертировать результат")
        mode = st.radio("Адаптивный алгоритм",
                        ["Блоки + Оцу (разбиение на подизображения)",
                         "Локальные статистики (среднее и СКО в окне)"])
        if mode.startswith("Блоки"):
            block = st.slider("Размер блока", 16, 256, 64, step=8)
            min_std = st.slider("Мин. СКО блока", 0, 40, 8)
        else:
            radius = st.slider("Радиус окна", 3, 60, 20)
            a = st.slider("a (порог по СКО: v > a·σ)", 0.0, 2.0, 0.0, step=0.05)
            b = st.slider("b (порог по среднему: v > b·m)", 0.5, 1.3, 0.9, step=0.01)

    # Источник: свой файл (эталона нет) или тестовое изображение (эталон известен)
    if upload:
        gray, truth = to_gray(Image.open(upload)), None
    else:
        gray, truth = test_images.make(name)
    gray = th.smooth(gray, smooth_r)               # необязательное подавление шума
    hist = th.histogram(gray)

    def acc(out):
        """Точность относительно эталона (если он есть). При инверсии эталон тоже инвертируется."""
        if truth is None:
            return ""
        t = ~truth if invert else truth
        return f", точность {((out == 255) == t).mean() * 100:.1f}%"

    t_otsu, ms1 = timed(th.otsu, hist)
    t_iter, ms2 = timed(th.iterative, hist)
    o_img, i_img = th.apply_global(gray, t_otsu, invert), th.apply_global(gray, t_iter, invert)
    if mode.startswith("Блоки"):
        (a_img, t_map), ms3 = timed(th.adaptive, gray, block, min_std, invert)
        a_cap = f"Адаптивный (блоки {block}, СКО ≥ {min_std}, {ms3:.1f} мс)"
    else:
        (a_img, t_map), ms3 = timed(th.adaptive_local, gray, radius, a, b, invert)
        a_cap = f"Адаптивный (окно {2 * radius + 1}, a={a}, b={b}, {ms3:.1f} мс)"

    c = st.columns(3)
    c[0].image(gray, caption="Оригинал (градации серого)", width="stretch")
    c[1].image(o_img, caption=f"Оцу: T = {t_otsu} ({ms1:.1f} мс){acc(o_img)}", width="stretch")
    c[2].image(i_img, caption=f"Итеративный: T = {t_iter} ({ms2:.1f} мс){acc(i_img)}", width="stretch")
    c = st.columns(3)
    c[0].image(a_img, caption=a_cap + acc(a_img), width="stretch")
    c[1].image(t_map, caption="Карта порогов адаптивного метода (ярче — выше порог)", width="stretch")
    c[2].pyplot(histogram.plot(hist, [(t_otsu, "#d9730d", "Оцу"), (t_iter, "#1f6fd0", "Итеративный")]))
    if truth is not None:
        st.caption("«Точность» — доля пикселей, совпавших с эталонной разметкой тестового изображения.")


# ---------------------------- страница 2: поэлементные операции и контраст ----------------------------
def page_pointwise(upload):
    st.header("Поэлементные операции и линейное контрастирование")
    with st.sidebar:
        name = st.selectbox("Тестовое изображение", list(test_images.CONTRAST))
    gray = to_gray(Image.open(upload)) if upload else test_images.make_contrast(name)

    tab1, tab2 = st.tabs(["Поэлементные операции", "Линейное контрастирование"])

    with tab1:
        kind = st.radio("Тип операции", ["Над одним изображением", "Над двумя изображениями"],
                        horizontal=True)
        if kind.startswith("Над одним"):
            op = st.selectbox("Операция", list(pointwise.OPS))
            fn, spec = pointwise.OPS[op]
            value = st.slider(*spec) if spec else None   # spec = (подпись, мин, макс, по умолч., шаг)
            result = fn(gray, value)
            note = ""
        else:
            op = st.selectbox("Операция", list(pointwise.OPS2))
            second = st.selectbox("Второе изображение", list(test_images.CONTRAST), index=2)
            b = test_images.make_contrast(second)
            if b.shape != gray.shape:                    # приводим к размеру первого
                b = np.asarray(Image.fromarray(b).resize(gray.shape[::-1]))
            result = pointwise.OPS2[op](gray, b)
            note = f"Второе изображение: {second}"
        c = st.columns(2)
        c[0].image(gray, caption="До: " + stats(gray), width="stretch")
        c[1].image(result, caption=f"{op}: " + stats(result), width="stretch")
        st.pyplot(histogram.plot_pair(th.histogram(gray), th.histogram(result)))
        if note:
            st.caption(note)

    with tab2:
        p = st.slider("Отсечение процентилей p, % (0 = классический min–max)", 0.0, 10.0, 0.0, step=0.5)
        result, lo, hi = pointwise.linear_contrast(gray, p)
        st.markdown(f"Диапазон [{lo:.0f}, {hi:.0f}] растягивается на [0, 255]: "
                    f"g = (v − {lo:.0f}) · 255 / {hi - lo:.0f}")
        c = st.columns(2)
        c[0].image(gray, caption="До: " + stats(gray), width="stretch")
        c[1].image(result, caption="После: " + stats(result), width="stretch")
        st.pyplot(histogram.plot_pair(th.histogram(gray), th.histogram(result)))


# ---------------------------- страница 3: пояснения ----------------------------
def page_report():
    st.header("Пояснения и анализ результатов")
    st.markdown((Path(__file__).parent / "report.md").read_text(encoding="utf-8"))


# ---------------------------- главное меню ----------------------------
st.title("Обработка цифровых изображений")
with st.sidebar:
    page = st.radio("Раздел", ["Пороговая обработка", "Поэлементные операции и контраст", "Пояснения"])
    upload = st.file_uploader("Своё изображение", type=["png", "jpg", "jpeg", "bmp", "tif"])
    st.divider()

if page == "Пороговая обработка":
    page_threshold(upload)
elif page == "Поэлементные операции и контраст":
    page_pointwise(upload)
else:
    page_report()
