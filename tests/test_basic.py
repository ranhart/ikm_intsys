import pytest
from app import predict_insurance_cost, demo

def test_prediction_runs():
    """Тест 1: Проверяем, что основная функция не падает на корректном примере."""
    try:
        result = predict_insurance_cost(
            age=30, 
            sex='Мужчина (Male)', 
            bmi=25.0, 
            children=0, 
            smoker='Нет (No)', 
            region='Юго-Запад (Southwest)'
        )
        assert result is not None
    except Exception as e:
        pytest.fail(f"Функция упала с ошибкой: {e}")

def test_prediction_format():
    """Тест 2: Проверяем, что функция возвращает правильный формат (строка с '$')."""
    result = predict_insurance_cost(
        age=28, 
        sex='Женщина (Female)', 
        bmi=22.5, 
        children=1, 
        smoker='Да (Yes)', 
        region='Северо-Восток (Northeast)'
    )
    # Проверяем, что результат — это текст (строка)
    assert isinstance(result, str)
    # Проверяем, что результат начинается со знака доллара
    assert result.startswith("$")

def test_app_launch():
    """Тест 3: Проверяем, что веб-приложение успешно собирается и готово к запуску."""
    assert demo is not None
    assert hasattr(demo, 'launch')