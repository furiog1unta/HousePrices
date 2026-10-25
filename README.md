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
│   └── 03_model_selection.ipynb
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

**03_model_selection**: сравнение моделей на 5-fold кросс-валидации (`KFold`, `shuffle=True`) и подбор гиперпараметров через `RandomizedSearchCV`.

## Результаты

RMSE на кросс-валидации, таргет в логарифмах:

| Модель | По умолчанию | После подбора |
|---|---|---|
| XGBoost | 0.1513 | **0.1318** |
| CatBoost | 0.1320 | 0.1322 |
| Lasso | 0.3988 | 0.1346 |
| LightGBM | 0.1415 | 0.1355 |
| Ridge | 0.1365 | 0.1364 |
| LinearRegression | 0.1399 | — |
| RandomForest | 0.1481 | 0.1428 |
| DecisionTree | 0.2060 | 0.1853 |


## Запуск

Нужен Python 3.14.

```bash
pip install -r requirements.txt
```

Ноутбуки запускаются по порядку: `01` → `02` → `03`. Ноутбук `02` создаёт `data/processed/`, а `03` его использует.
