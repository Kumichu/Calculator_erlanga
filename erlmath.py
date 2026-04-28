import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def erlang_a_calculator(
        lambda_: float,
        mu: float,
        c: int,
        K: int,
        sigma: float
) -> Dict[str, float]:
    """
    Расчёт показателей для модели Эрланга A (M/M/c/с+K) с конечной очередью
    и экспоненциальным временем терпения.

    Параметры:
    ----------
    lambda_ : float
        Интенсивность поступления заявок (ед./время)
    mu : float
        Интенсивность обслуживания одним каналом (ед./время)
    c : int
        Число обслуживающих каналов (серверов)
    K : int
        Общее число мест в системе (c + размер очереди). K >= c >= 1
    sigma : float
        Интенсивность ухода из очереди одного ожидающего клиента (ед./время)

    Возвращает:
    ----------
    Dict[str, float]
        Словарь с основными характеристиками системы:
        - P_block: вероятность блокировки (потери) заявки
        - P_wait: вероятность ожидания для поступающей заявки
        - P_immediate: вероятность немедленного обслуживания
        - P_abandon: доля заявок, покинувших очередь из-за нетерпения
        - L_q: средняя длина очереди
        - L_s: среднее число занятых каналов
        - L: среднее число заявок в системе
        - W_q: среднее время ожидания в очереди (для всех принятых заявок)
        - W: среднее время пребывания в системе (для всех принятых заявок)
        - W_q_given_wait: среднее время ожидания для тех, кто встал в очередь
        - W_q_served: среднее время ожидания для обслуженных заявок
        - lambda_eff: эффективная интенсивность входящего потока
        - rho: нагрузка на один канал (λ/(cμ))
    """
    # Проверка входных данных
    if lambda_ <= 0 or mu <= 0 or sigma < 0:
        raise ValueError("Интенсивности должны быть положительными")
    if c <= 0 or K < c:
        raise ValueError("Должно быть c >= 1 и K >= c")

    # Число каналов небольшое - применяем прямое вычисление
    # if c <= 90:
    try:
        # Вычисление относительных вероятностей (p_n / p_0)
        rel_probs = [0.0] * (K + 1)

        # Часть для n = 0,1,...,c
        for n in range(c + 1):
            rel_probs[n] = (lambda_ / mu) ** n / math.factorial(n)

        # Часть для n = c+1, ..., K
        if K > c:
            p_c_rel = rel_probs[c]
            prod = 1.0
            for i in range(1, K - c + 1):
                prod *= lambda_ / (c * mu + i * sigma)
                rel_probs[c + i] = p_c_rel * prod

        # Нормировочная константа
        sum_probs = sum(rel_probs)
        if sum_probs == 0:
            raise ValueError("Сумма вероятностей равна нулю, проверьте параметры")

        # Вероятность пустого состояния
        p0 = 1.0 / sum_probs

        # Абсолютные вероятности состояний
        p = [p0 * rp for rp in rel_probs]
    # Число каналов большое - применяем рекуррентные формулы в логарифмической шкале
    # else:
    except OverflowError:
        # Логарифмы относительных вероятностей ln(p_n / p_0)
        ln_rel = [0.0] * (K + 1)

        # Рекуррентное заполнение для n = 1..c
        for n in range(1, c + 1):
            ln_rel[n] = ln_rel[n - 1] + math.log(lambda_) - math.log(n) - math.log(mu)

        # Рекуррентное заполнение для n = c+1..K
        for n in range(c + 1, K + 1):
            i = n - c  # номер места в очереди (i >= 1)
            denom = c * mu + i * sigma
            if denom <= 0:
                raise ValueError(f"Знаменатель {denom} <= 0 при n={n}")
            ln_rel[n] = ln_rel[n - 1] + math.log(lambda_) - math.log(denom)

        # Устойчивое вычисление суммы exp(ln_rel) через log-sum-exp
        max_ln = max(ln_rel)
        sum_exp = sum(math.exp(ln - max_ln) for ln in ln_rel)
        log_sum = max_ln + math.log(sum_exp)

        # Вероятность пустого состояния
        p0 = math.exp(-log_sum)

        # Абсолютные вероятности состояний
        p = [math.exp(ln - log_sum) for ln in ln_rel]

    # Расчёт основных показателей
    # Среднее число занятых каналов
    L_s = sum(n * p[n] for n in range(c + 1)) + c * sum(p[n] for n in range(c + 1, K + 1))
    # Вероятность блокировки (потери) заявки
    P_block = p[K] if K < len(p) else 0.0
    # Эффективная интенсивность входящего потока (принятые заявки)
    lambda_eff = lambda_ * (1.0 - P_block)
    # Параметр нагрузки на один канал
    rho = lambda_ / (c * mu)

    if c != K:
        # Средняя длина очереди
        L_q = sum((n - c) * p[n] for n in range(c + 1, K + 1))
        # Среднее число заявок в системе
        L = L_q + L_s
        # Вероятность ожидания (заявка застаёт все каналы занятыми и есть место в очереди)
        P_wait = sum(p[n] for n in range(c, K))  # при n = K заявка блокируется, не ждёт
        # Вероятность немедленного обслуживания
        P_immediate = abs(1.0 - P_block - P_wait)
        # Среднее время ожидания в очереди для всех принятых заявок
        W_q = L_q / lambda_eff if lambda_eff > 0 else 0.0
        # Среднее время пребывания в системе для всех принятых заявок
        W = L / lambda_eff if lambda_eff > 0 else 0.0
        # Вероятность ухода из очереди из-за нетерпения (доля от всех поступивших)
        P_abandon = (sigma * L_q) / lambda_ if lambda_ > 0 else 0.0
        # Доля необслуженных заявок, сумма ушедших из очереди и непоступивших при полной занятости
        P_out = P_block + P_abandon
        # Эффективная интенсивность входящего потока (принятые заявки, с учётом нетерпения)
        lambda_eff = lambda_ * (1.0 - P_out)
        # Среднее время ожидания для тех, кто встал в очередь (включая ушедших)
        W_q_given_wait = L_q / (lambda_ * P_wait) if P_wait > 0 else 0.0
        # Среднее время ожидания для обслуженных заявок (приближённая оценка через Литтла)
        lambda_served = lambda_ * (1.0 - P_block - P_abandon)
        W_q_served = L_q / lambda_served if lambda_served > 0 else 0.0
        return {
            "P_block": P_block,  # Вероятность блокировки (потери) заявки
            "P_wait": P_wait,  # Вероятность ожидания (заявка застаёт все каналы занятыми и есть место в очереди)
            "P_immediate": P_immediate,  # Вероятность немедленного обслуживания
            "P_abandon": P_abandon,  # Вероятность ухода из очереди из-за нетерпения (доля от всех поступивших)
            "P_out": P_out, # Доля необслуженных заявок
            "L_q": L_q,  # Средняя длина очереди
            "L_s": L_s,  # Среднее число занятых каналов
            "L": L,  # Среднее число заявок в системе
            "W_q": W_q,  # Среднее время ожидания в очереди для всех принятых заявок
            "W": W,  # Среднее время пребывания в системе для всех принятых заявок
            "W_q_given_wait": W_q_given_wait,  # Среднее время ожидания для тех, кто встал в очередь (включая ушедших)
            "W_q_served": W_q_served, # Среднее время ожидания для обслуженных заявок
            "lambda_eff": lambda_eff, # Эффективная интенсивность входящего потока (принятые заявки)
            "rho": rho  # Параметр нагрузки на один канал
        }
    else:
        # Случай K == c: очереди нет, поэтому все "очередные" метрики равны 0
        return {
            "P_block": P_block,  # Вероятность блокировки (потери) заявки
            # "P_wait": 0.0,  # Ожидания нет, так как очередь отсутствует
            "P_immediate": 1.0 - P_block,  # Все принятые заявки обслуживаются сразу
            # "P_abandon": 0.0,  # Ухода из очереди нет, так как очереди нет
            "P_out": P_block, # Доля необслуженных заявок
            # "L_q": 0.0,  # Средняя длина очереди
            "L_s": L_s,  # Среднее число занятых каналов
            "L": L_s,  # Среднее число заявок в системе = число занятых каналов
            # "W_q": 0.0,  # Среднее время ожидания в очереди
            "W": (L_s / lambda_eff) if lambda_eff > 0 else 0.0,  # Среднее время пребывания в системе
            # "W_q_given_wait": 0.0,  # Никто не ожидает
            # "W_q_served": 0.0,  # Для обслуженных ожидание отсутствует
            "lambda_eff": lambda_eff,  # Эффективная интенсивность входящего потока (принятые заявки)
            "rho": rho  # Параметр нагрузки на один канал
        }


def erlang_b_calculator(
        lambda_: float,
        mu: float,
        c: int
) -> Dict[str, float]:
    return erlang_a_calculator(lambda_, mu, c, c, 0)


def erlang_c_calculator(
        lambda_: float,
        mu: float,
        c: int,
        K: int
) -> Dict[str, float]:
    return erlang_a_calculator(lambda_, mu, c, K, 0)


def erlang_a_find_c(
    lambda_: float,
    mu: float,
    K_delta: int,
    sigma: float,
    P_out_t: float
) -> Dict[str, float]:
    """
    Подбирает минимально возможное число каналов c.

    Параметры:
    ----------
    lambda_ : float
        Интенсивность поступления заявок (ед./время)
    mu : float
        Интенсивность обслуживания одним каналом (ед./время)
    K_delta : int
        Размер очереди
    sigma : float
        Интенсивность ухода из очереди одного ожидающего клиента (ед./время)
    P_out_t : float
        Требуемое значение P_out

    Возвращает:
    ----------
    Dict[str, float]
        Словарь с основными характеристиками системы:
        - То же, что и erlang_a_calculator
        - Значение c
    """
    c = 0
    while True:
        if c<20:
            c+=1
        elif c<200:
            c+=2
        elif c<300:
            c+=5
        else:
            c+=10
        res = erlang_a_calculator(lambda_, mu, c, c+K_delta, sigma)
        P_out = res.get("P_out", 1)
        if P_out<=P_out_t:
            res["c"] = c
            return res
        if c>10**5:
            raise ValueError("Число каналов выходит больше 10^5")


def erlang_a_find_lambda(
    mu: float,
    c: int,
    K: int,
    sigma: float,
    P_out_t: float
) -> Dict[str, float]:
    """
    Подбирает максимально возможное значение lambda.

    Параметры:
    ----------
    mu : float
        Интенсивность обслуживания одним каналом (ед./время)
    c : int
        Число обслуживающих каналов (серверов)
    K : int
        Общее число мест в системе (c + размер очереди). K >= c >= 1
    sigma : float
        Интенсивность ухода из очереди одного ожидающего клиента (ед./время)
    P_out_t : float
        Требуемое значение P_out

    Возвращает:
    ----------
    Dict[str, float]
        Словарь с основными характеристиками системы:
        - То же, что и erlang_a_calculator
        - Значение lambda
    """
    lambda_ = 0
    lambda_prev = 0
    while True:
        lambda_prev = lambda_
        if lambda_<50:
            lambda_+=0.1
        elif lambda_<200:
            lambda_+=1
        else:
            lambda_+=5
        res = erlang_a_calculator(lambda_, mu, c, K, sigma)
        P_out = res.get("P_out", 1)
        if P_out>P_out_t:
            res = erlang_a_calculator(lambda_prev, mu, c, K, sigma)
            res["lambda"] = lambda_prev
            return res
        if lambda_>5000:
            raise ValueError("lambda выходит больше 5000")


def plot_erlang_a_analysis_by_lambda_OLD(
        lambda_range: Tuple[float, float],
        mu: float,
        c: int,
        K: int,
        sigma: float
) -> None:
    """
    Построение графиков зависимости основных характеристик от интенсивности λ.

    Параметры:
    ----------
    lambda_range : Tuple[float, float]
        Диапазон изменения λ (min, max)
    mu, c, K, theta : параметры модели (константы)
    num_points : int
        Количество точек для построения графиков
    """
    lambdas = np.linspace(lambda_range[0], lambda_range[1], 100)
    results = {
        "P_block": [],
        "P_abandon": [],
        "L_q": [],
        "W_q": [],
        "rho": []
    }

    for lam in lambdas:
        try:
            res = erlang_a_calculator(lam, mu, c, K, sigma)
            results["P_block"].append(res["P_block"])
            results["P_abandon"].append(res["P_abandon"])
            results["L_q"].append(res["L_q"])
            results["W_q"].append(res["W_q"])
            results["rho"].append(res["rho"])
        except ValueError:
            # Если при каком-то λ возникает ошибка (например, ρ > 1), пропускаем
            for key in results:
                results[key].append(np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Модель Эрланга A: c={c}, K={K}, μ={mu:.2f}, θ={sigma:.2f}', fontsize=14)

    # Вероятность блокировки
    ax = axes[0, 0]
    ax.plot(lambdas, results["P_block"], 'b-', linewidth=2)
    ax.set_xlabel('Интенсивность поступления λ')
    ax.set_ylabel('P_block')
    ax.grid(True, alpha=0.3)

    # Вероятность ухода из очереди
    ax = axes[0, 1]
    ax.plot(lambdas, results["P_abandon"], 'r-', linewidth=2)
    ax.set_xlabel('Интенсивность поступления λ')
    ax.set_ylabel('P_abandon')
    ax.grid(True, alpha=0.3)

    # Средняя длина очереди
    ax = axes[1, 0]
    ax.plot(lambdas, results["L_q"], 'g-', linewidth=2)
    ax.set_xlabel('Интенсивность поступления λ')
    ax.set_ylabel('L_q (средняя длина очереди)')
    ax.grid(True, alpha=0.3)

    # Среднее время ожидания в очереди
    ax = axes[1, 1]
    ax.plot(lambdas, results["W_q"], 'm-', linewidth=2)
    ax.set_xlabel('Интенсивность поступления λ')
    ax.set_ylabel('W_q (среднее время ожидания)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_erlang_a_analysis_by_c_OLD(
        lam: float,
        mu: float,
        c_range: Tuple[int, int],
        K_delta: int,
        sigma: float
) -> None:
    """
    Построение графиков зависимости основных характеристик от интенсивности λ.

    Параметры:
    ----------
    lambda_range : Tuple[float, float]
        Диапазон изменения λ (min, max)
    mu, c, K, theta : параметры модели (константы)
    num_points : int
        Количество точек для построения графиков
    """
    cs = range(c_range[0], c_range[1] + 1)
    results = {
        "P_block": [],
        "P_abandon": [],
        "L_q": [],
        "W_q": [],
        "rho": []
    }
    for c in cs:
        res = erlang_a_calculator(lam, mu, c, c + K_delta, sigma)
        results["P_block"].append(res["P_block"])
        results["P_abandon"].append(res["P_abandon"])
        results["L_q"].append(res["L_q"])
        results["W_q"].append(res["W_q"])
        results["rho"].append(res["rho"])
        # try:

        # except ValueError:
        #     # Если при каком-то λ возникает ошибка (например, ρ > 1), пропускаем
        #     for key in results:
        #         results[key].append(np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Модель Эрланга A: lambda={lam}, K_delta={K_delta}, μ={mu:.2f}, θ={sigma:.2f}', fontsize=14)

    # Вероятность блокировки
    ax = axes[0, 0]
    ax.plot(cs, results["P_block"], 'b-', linewidth=2)
    ax.set_xlabel('Число каналов c')
    ax.set_ylabel('P_block')
    ax.grid(True, alpha=0.3)

    # Вероятность ухода из очереди
    ax = axes[0, 1]
    ax.plot(cs, results["P_abandon"], 'r-', linewidth=2)
    ax.set_xlabel('Число каналов c')
    ax.set_ylabel('P_abandon')
    ax.grid(True, alpha=0.3)

    # Средняя длина очереди
    ax = axes[1, 0]
    ax.plot(cs, results["L_q"], 'g-', linewidth=2)
    ax.set_xlabel('Число каналов c')
    ax.set_ylabel('L_q (средняя длина очереди)')
    ax.grid(True, alpha=0.3)

    # Среднее время ожидания в очереди
    ax = axes[1, 1]
    ax.plot(cs, results["W_q"], 'm-', linewidth=2)
    ax.set_xlabel('Число каналов c')
    ax.set_ylabel('W_q (среднее время ожидания)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_erlang_a_by_lambda(
        lambda_range: Tuple[float, float],
        mu: float,
        c: int,
        K: int,
        sigma: float,
        num_points: int = 100
) -> str:
    lambdas = np.linspace(lambda_range[0], lambda_range[1], num_points).tolist()
    results = {
        "P_block": [],
        "P_abandon": [],
        "L_q": [],
        "W_q": [],
        "rho": []
    }

    for lam in lambdas:
        res = erlang_a_calculator(lam, mu, c, K, sigma)
        results["P_block"].append(res["P_block"])
        results["P_abandon"].append(res["P_abandon"])
        results["L_q"].append(res["L_q"])
        results["W_q"].append(res["W_q"])
        results["rho"].append(res["rho"])

    # Создаём сетку 2x2 с общими настройками
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Вероятность блокировки",
            "Вероятность ухода из очереди",
            "Средняя длина очереди (L_q)",
            "Среднее время ожидания (W_q)"
        ),
        row_heights=[0.5, 0.5],  # Относительная высота строк (50% и 50%)
        # Можно задать абсолютную высоту в пикселях, но row_heights работает с относительными значениями
        vertical_spacing=0.2,  # Расстояние между строками (по умолчанию 0.1)
        horizontal_spacing=0.1  # Расстояние между колонками
    )

    fig.update_layout(
        title_text=f'Модель Эрланга A: c={c}, K={K}, μ={mu:.2f}, σ={sigma:.2f}',
        title_x=0.5,  # центрируем заголовок
        showlegend=False,  # легенда не нужна, т.к. подписи уже в заголовках subplot
        height=800
        # width=800
    )

    fig.add_trace(
        go.Scatter(x=lambdas, y=results["P_block"], mode='lines', name='P_block',
                   line=dict(color='blue', width=2)), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=lambdas, y=results["P_abandon"], mode='lines', name='P_abandon',
                   line=dict(color='red', width=2)), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=lambdas, y=results["L_q"], mode='lines', name='L_q',
                   line=dict(color='green', width=2)), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=lambdas, y=results["W_q"], mode='lines', name='W_q',
                   line=dict(color='magenta', width=2)), row=2, col=2
    )

    fig.update_xaxes(title_text="Интенсивность поступления λ", row=1, col=1)
    fig.update_xaxes(title_text="Интенсивность поступления λ", row=1, col=2)
    fig.update_xaxes(title_text="Интенсивность поступления λ", row=2, col=1)
    fig.update_xaxes(title_text="Интенсивность поступления λ", row=2, col=2)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig.to_html(full_html=False, include_plotlyjs=False)

def plot_erlang_a_by_c(
        lam: float,
        mu: float,
        c_range: Tuple[int, int],
        K_delta: int,
        sigma: float
) -> str:
    cs = range(c_range[0], c_range[1] + 1)
    results = {
        "P_block": [],
        "P_abandon": [],
        "L_q": [],
        "W_q": [],
        "rho": []
    }
    for c in cs:
        res = erlang_a_calculator(lam, mu, c, c + K_delta, sigma)
        results["P_block"].append(res["P_block"])
        results["P_abandon"].append(res["P_abandon"])
        results["L_q"].append(res["L_q"])
        results["W_q"].append(res["W_q"])
        results["rho"].append(res["rho"])
    cs = np.linspace(c_range[0], c_range[1], num=c_range[1]-c_range[0]+1).tolist()

    # Создаём сетку 2x2 с общими настройками
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Вероятность блокировки",
            "Вероятность ухода из очереди",
            "Средняя длина очереди (L_q)",
            "Среднее время ожидания (W_q)"
        ),
        row_heights=[0.5, 0.5],  # Относительная высота строк (50% и 50%)
        # Можно задать абсолютную высоту в пикселях, но row_heights работает с относительными значениями
        vertical_spacing=0.2,  # Расстояние между строками (по умолчанию 0.1)
        horizontal_spacing=0.1  # Расстояние между колонками
    )

    fig.update_layout(
        title_text=f'Модель Эрланга A: lambda={lam}, K_delta={K_delta}, μ={mu:.2f}, σ={sigma:.2f}',
        title_x=0.5,  # центрируем заголовок
        showlegend=False,  # легенда не нужна, т.к. подписи уже в заголовках subplot
        height=800
        # width=800
    )

    fig.add_trace(
        go.Scatter(x=cs, y=results["P_block"], mode='lines', name='P_block',
                   line=dict(color='blue', width=2)), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=cs, y=results["P_abandon"], mode='lines', name='P_abandon',
                   line=dict(color='red', width=2)), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=cs, y=results["L_q"], mode='lines', name='L_q',
                   line=dict(color='green', width=2)), row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=cs, y=results["W_q"], mode='lines', name='W_q',
                   line=dict(color='magenta', width=2)), row=2, col=2
    )

    fig.update_xaxes(title_text="Число каналов c", row=1, col=1)
    fig.update_xaxes(title_text="Число каналов c", row=1, col=2)
    fig.update_xaxes(title_text="Число каналов c", row=2, col=1)
    fig.update_xaxes(title_text="Число каналов c", row=2, col=2)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig.to_html(full_html=False, include_plotlyjs=False)
# -------------------------------------------------------------------
# Пример использования
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Параметры колл-центра
    lambda_ = 120.0  # 170 звонков в минуту
    mu = 1 / 1  # среднее время разговора 4 минуты (μ = 1/4)
    c = 100  # 600 операторов
    K = c + 15  # 600+100 мест в системе (100 мест в очереди)
    # K = c+0      # 600+0 мест в системе (0 мест в очереди)
    sigma = 4 / 5  # среднее терпение 15 минут (θ = 1/15)

    print(erlang_a_find_c(lambda_, mu, K-c, sigma, 0.05))
    print(erlang_a_find_lambda(mu, c, K, sigma, 0.05))
    quit()
    print("Расчёт характеристик для заданных параметров:\n")
    res = erlang_a_calculator(lambda_, mu, c, K, sigma)
    for key, value in res.items():
        if key in ["P_block", "P_wait", "P_immediate", "P_abandon"]:
            print(f"{key:20s}: {value:.6f} ({value * 100:.2f} %)")
        else:
            print(f"{key:20s}: {value:.4f}")

    if c != K:
        # Построение графиков зависимости от λ в диапазоне [600, 1000]
        plot_erlang_a_by_lambda((lambda_ * 0.5, lambda_ * 1.5), mu, c, K, sigma, num_points=100)
        plot_erlang_a_by_c(lambda_, mu, ((c-20 if c>20 else 1), c+20), K - c, sigma)

# Расчёт показателей для модели Эрланга A (M/M/c/K+M) с конечной очередью
# и экспоненциальным временем терпения.

# Параметры:
# ----------
# lambda_ : float
#     Интенсивность поступления заявок (ед./время)
# mu : float
#     Интенсивность обслуживания одним каналом (ед./время)
# c : int
#     Число обслуживающих каналов (серверов)
# K : int
#     Общее число мест в системе (c + размер очереди). K >= c >= 1
# theta : float
#     Интенсивность ухода из очереди одного ожидающего клиента (ед./время)

# Возвращает:
# ----------
# Dict[str, float]
#     Словарь с основными характеристиками системы:
#     - P_block: вероятность блокировки (потери) заявки
#     - P_wait: вероятность ожидания для поступающей заявки
#     - P_immediate: вероятность немедленного обслуживания
#     - P_abandon: доля заявок, покинувших очередь из-за нетерпения
#     - L_q: средняя длина очереди
#     - L_s: среднее число занятых каналов
#     - L: среднее число заявок в системе
#     - W_q: среднее время ожидания в очереди (для всех принятых заявок)
#     - W: среднее время пребывания в системе (для всех принятых заявок)
#     - W_q_given_wait: среднее время ожидания для тех, кто встал в очередь
#     - W_q_served: среднее время ожидания для обслуженных заявок
#     - lambda_eff: эффективная интенсивность входящего потока
#     - rho: нагрузка на один канал (λ/(cμ))