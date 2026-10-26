# HousePrices

Предсказание цены дома по его характеристикам: учебный проект по соревнованию Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

- **Задача:** регрессия, таргет `SalePrice`
- **Данные:** 1460 домов в train, 1459 в test, 79 признаков
- **Метрика:** RMSE между логарифмами предсказанной и реальной цены (метрика соревнования)

## Структура

```
HousePrices/
├── data/
│   ├── raw/          # исходные данные Kaggle + data_description.txt
│   └── processed/    # данные после обработки (создаются в 02_data_processing)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_data_processing.ipynb
│   ├── 03_model_selection.ipynb
│   └── 04_final_submission.ipynb
├── submissions/      # файлы с предсказаниями для отправки на Kaggle
└── requirements.txt
```

## Ноутбуки

**01_EDA**: разведочный анализ. Распределение таргета, корреляции с ценой, мультиколлинеарные пары признаков (`GarageArea`/`GarageCars`, `GrLivArea`/`TotRmsAbvGrd`, `TotalBsmtSF`/`1stFlrSF`), признаки, где пропусков больше 70%.

**02_data_processing**: подготовка данных.
1. Удаление признаков, слабо связанных с ценой, сильно коррелирующих с другими или почти пустых
2. Заполнение пропусков: `'None'` или `0` там, где NaN означает отсутствие объекта (нет гаража, подвала и т.п.); остальное заполняется по статистикам train
3. Ordinal-кодирование оценок качества (`Po < Fa < TA < Gd < Ex`)
4. One-hot кодирование номинальных признаков, выравнивание колонок test по train
5. `log1p` для таргета (`LogSalePrice`) и признаков со скошенным распределением
6. Стандартизация небинарных признаков по `mean`/`std` из train

**03_model_selection**: сравнение восьми моделей (линейные, деревья, бустинги) на 5-fold кросс-валидации (`KFold`, `shuffle=True`) и грубый подбор гиперпараметров через `RandomizedSearchCV`. Для дальнейшей работы отобраны три лучшие модели: XGBoost, CatBoost и Lasso.

**04_final_submission**: подбор гиперпараметров для XGBoost, CatBoost и Lasso с помощью Optuna (TPE-сэмплер, 50 trials на модель, та же 5-fold кросс-валидация). Лучшая модель, CatBoost, обучается на всём train, предсказания переводятся из логарифмов обратно через `expm1` и сохраняются в `submissions/submissioncatboost.csv`.

## Результаты

RMSE после подбора гиперпараметров в Optuna, таргет в логарифмах. **CV** — 5-fold кросс-валидация на train, **LB** — public leaderboard Kaggle:

| Модель | CV | LB |
|---|---|---|
| CatBoost | **0.13200** | **0.12901** |
| XGBoost | 0.13201 | — |
| Lasso | 0.13447 | — |

На Kaggle отправлялись только предсказания CatBoost. На leaderboard модель показала результат чуть лучше, чем на кросс-валидации.

Лучшие гиперпараметры:

- **CatBoost:** `iterations=500`, `learning_rate≈0.073`, `depth=6`, `l2_leaf_reg≈3.68`, `random_strength≈0.59`, `bagging_temperature≈0.66`
- **XGBoost:** `n_estimators=700`, `learning_rate≈0.040`, `max_depth=3`, `min_child_weight=2`, `subsample≈0.57`, `colsample_bytree≈0.27`, `reg_alpha≈0.15`, `reg_lambda≈1.2e-4`
- **Lasso:** `alpha≈4.05e-4`

## Запуск

Нужен Python 3.14.

```bash
pip install -r requirements.txt
```

Ноутбуки запускаются по порядку: `01` → `02` → `03` → `04`. Ноутбук `02` создаёт `data/processed/`, а `03` и `04` его используют.
