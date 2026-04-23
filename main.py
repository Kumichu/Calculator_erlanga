from flask import Flask, render_template, request
import webbrowser
import threading
import os
from typing import Dict

from erlmath import erlang_a_calculator, plot_erlang_a_by_lambda, plot_erlang_a_by_c

app = Flask(__name__)


def erlang_b_calculator(lambda_: float, mu: float, c: int) -> Dict[str, float]:
    """
    Эрланг B (M/M/c/c): система с отказами без очереди
    """
    if lambda_ <= 0 or mu <= 0:
        raise ValueError("Интенсивности должны быть положительными")
    if c <= 0:
        raise ValueError("Число каналов должно быть >= 1")

    a = lambda_ / mu

    inv_b = 1.0
    for n in range(1, c + 1):
        inv_b = 1.0 + inv_b * n / a
    p_block = 1.0 / inv_b

    lambda_eff = lambda_ * (1.0 - p_block)
    l_s = a * (1.0 - p_block)
    rho = lambda_ / (c * mu)

    return {
        "P_block": p_block,
        "P_immediate": 1.0 - p_block,
        "L_s": l_s,
        "lambda_eff": lambda_eff,
        "rho": rho,
    }


def build_result_rows(metrics: Dict[str, float]):
    labels = {
        "P_block": "Вероятность блокировки",
        "P_wait": "Вероятность ожидания",
        "P_immediate": "Вероятность немедленного обслуживания",
        "P_abandon": "Доля ушедших из очереди",
        "L_q": "Средняя длина очереди",
        "L_s": "Среднее число занятых каналов",
        "L": "Среднее число заявок в системе",
        "W_q": "Среднее время ожидания в очереди",
        "W": "Среднее время пребывания в системе",
        "W_q_given_wait": "Среднее ожидание для ожидавших",
        "W_q_served": "Среднее ожидание для обслуженных",
        "lambda_eff": "Эффективная интенсивность потока",
        "rho": "Нагрузка на один канал",
    }

    percent_keys = {"P_block", "P_wait", "P_immediate", "P_abandon"}

    rows = []
    for key, value in metrics.items():
        if key in percent_keys:
            formatted = f"{value:.6f} ({value * 100:.2f}%)"
        else:
            formatted = f"{value:.6f}"
        rows.append({
            "key": key,
            "label": labels.get(key, key),
            "value": formatted
        })
    return rows


def parse_erlang_a_args(args):
    lambda_ = float(args.get("lambda_", 600))
    mu = float(args.get("mu", 0.25))
    c = int(args.get("c", 400))
    K = int(args.get("K", 500))
    theta = float(args.get("theta", 0.2))

    if lambda_ <= 0 or mu <= 0 or theta < 0:
        raise ValueError("Интенсивности должны быть положительными, theta >= 0")
    if c <= 0 or K < c:
        raise ValueError("Должно быть c >= 1 и K >= c")

    return lambda_, mu, c, K, theta


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculator/erlanga")
def calculator_erlanga():
    return render_template("calculator_erlanga_a.html")


def render_erlanga_result_page(lambda_, mu, c, K, theta, metrics, mode="lambda"):
    if mode == "c":
        c_min = max(1, c - 20)
        c_max = max(c_min, c + 20)
        plot_div = plot_erlang_a_by_c(lambda_, mu, (c_min, c_max), K - c, theta)
        page_subtitle = "Зависимость метрик от числа каналов c"
        page_description = "Диапазон изменения c: от c-20 до c+20."
    else:
        mode = "lambda"
        plot_div = plot_erlang_a_by_lambda((lambda_ * 0.5, lambda_ * 1.5), mu, c, K, theta)
        page_subtitle = "Зависимость метрик от интенсивности поступления λ"
        page_description = "Диапазон изменения λ: от 0.5λ до 1.5λ."

    return render_template(
        "calculator_erlanga_graphs.html",
        plot_div=plot_div,
        results=build_result_rows(metrics),
        lambda_=lambda_,
        mu=mu,
        c=c,
        K=K,
        theta=theta,
        mode=mode,
        page_subtitle=page_subtitle,
        page_description=page_description
    )


@app.route("/calculator/erlanga/graphs")
def erlanga_graphs():
    try:
        lambda_, mu, c, K, theta = parse_erlang_a_args(request.args)
        mode = request.args.get("mode", "lambda")
        metrics = erlang_a_calculator(lambda_, mu, c, K, theta)
        return render_erlanga_result_page(lambda_, mu, c, K, theta, metrics, mode)
    except ValueError as e:
        return render_template(
            "calculator_erlanga_a.html",
            error=str(e),
            lambda_=request.args.get("lambda_", ""),
            mu=request.args.get("mu", ""),
            c=request.args.get("c", ""),
            K=request.args.get("K", ""),
            theta=request.args.get("theta", "")
        )
    except Exception as e:
        return render_template(
            "calculator_erlanga_a.html",
            error=f"Ошибка: {str(e)}",
            lambda_=request.args.get("lambda_", ""),
            mu=request.args.get("mu", ""),
            c=request.args.get("c", ""),
            K=request.args.get("K", ""),
            theta=request.args.get("theta", "")
        )


@app.route("/calculator/erlangb")
def calculator_erlangb():
    return render_template("calculator_erlanga_b.html")


@app.route("/calculator/erlangc")
def calculator_erlangc():
    return render_template("calculator_erlanga_c.html")


@app.route("/calculator/basic")
def calculator_basic():
    return render_template("calculator_basic.html")


@app.route("/calculator/basic/calculate", methods=["POST"])
def calculate_basic():
    try:
        num1 = float(request.form["num1"])
        num2 = float(request.form["num2"])
        operation = request.form["operation"]

        if operation == "add":
            result = num1 + num2
            operation_symbol = "+"
        elif operation == "subtract":
            result = num1 - num2
            operation_symbol = "-"
        elif operation == "multiply":
            result = num1 * num2
            operation_symbol = "×"
        elif operation == "divide":
            if num2 == 0:
                return render_template(
                    "calculator_basic.html",
                    error="Деление на ноль невозможно",
                    num1=num1,
                    num2=num2,
                    operation=operation
                )
            result = num1 / num2
            operation_symbol = "÷"
        else:
            return render_template(
                "calculator_basic.html",
                error="Неверная операция"
            )

        calculation = f"{num1} {operation_symbol} {num2} = {result}"
        return render_template(
            "calculator_basic.html",
            result=result,
            calculation=calculation,
            num1=num1,
            num2=num2,
            operation=operation
        )
    except ValueError:
        return render_template(
            "calculator_basic.html",
            error="Пожалуйста, введите корректные числа"
        )
    except Exception as e:
        return render_template(
            "calculator_basic.html",
            error=f"Ошибка: {str(e)}"
        )


@app.route("/calculator/erlanga/calculate", methods=["POST"])
def calculate_erlanga():
    try:
        lambda_ = float(request.form["lambda_"])
        mu = float(request.form["mu"])
        c = int(request.form["c"])
        K = int(request.form["K"])
        theta = float(request.form["theta"])

        metrics = erlang_a_calculator(lambda_, mu, c, K, theta)

        return render_erlanga_result_page(
            lambda_=lambda_,
            mu=mu,
            c=c,
            K=K,
            theta=theta,
            metrics=metrics,
            mode="lambda"
        )
    except ValueError as e:
        return render_template(
            "calculator_erlanga_a.html",
            error=str(e),
            lambda_=request.form.get("lambda_", ""),
            mu=request.form.get("mu", ""),
            c=request.form.get("c", ""),
            K=request.form.get("K", ""),
            theta=request.form.get("theta", "")
        )
    except Exception as e:
        return render_template(
            "calculator_erlanga_a.html",
            error=f"Ошибка: {str(e)}",
            lambda_=request.form.get("lambda_", ""),
            mu=request.form.get("mu", ""),
            c=request.form.get("c", ""),
            K=request.form.get("K", ""),
            theta=request.form.get("theta", "")
        )


@app.route("/calculator/erlangb/calculate", methods=["POST"])
def calculate_erlangb():
    try:
        lambda_ = float(request.form["lambda_"])
        mu = float(request.form["mu"])
        c = int(request.form["c"])

        metrics = erlang_b_calculator(lambda_, mu, c)

        return render_template(
            "calculator_erlanga_b.html",
            results=build_result_rows(metrics),
            lambda_=lambda_,
            mu=mu,
            c=c
        )
    except ValueError as e:
        return render_template(
            "calculator_erlanga_b.html",
            error=str(e),
            lambda_=request.form.get("lambda_", ""),
            mu=request.form.get("mu", ""),
            c=request.form.get("c", "")
        )
    except Exception as e:
        return render_template(
            "calculator_erlanga_b.html",
            error=f"Ошибка: {str(e)}",
            lambda_=request.form.get("lambda_", ""),
            mu=request.form.get("mu", ""),
            c=request.form.get("c", "")
        )


@app.route("/calculator/erlangc/calculate", methods=["POST"])
def calculate_erlangc():
    try:
        lambda_ = float(request.form["lambda_"])
        mu = float(request.form["mu"])
        c = int(request.form["c"])
        K = int(request.form["K"])

        metrics = erlang_a_calculator(lambda_, mu, c, K, 0)

        return render_template(
            "calculator_erlanga_c.html",
            results=build_result_rows(metrics),
            lambda_=lambda_,
            mu=mu,
            c=c,
            K=K
        )
    except ValueError as e:
        return render_template(
            "calculator_erlanga_c.html",
            error=str(e),
            lambda_=request.form.get("lambda_", ""),
            mu=request.form.get("mu", ""),
            c=request.form.get("c", ""),
            K=request.form.get("K", "")
        )
    except Exception as e:
        return render_template(
            "calculator_erlanga_c.html",
            error=f"Ошибка: {str(e)}",
            lambda_=request.form.get("lambda_", ""),
            mu=request.form.get("mu", ""),
            c=request.form.get("c", ""),
            K=request.form.get("K", "")
        )


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        def open_browser():
            webbrowser.open_new("http://127.0.0.1:5000/")

        threading.Timer(0, open_browser).start()

    app.run(debug=True)
