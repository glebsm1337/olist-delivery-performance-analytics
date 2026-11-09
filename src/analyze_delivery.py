"""Анализ сроков доставки на публичном наборе данных Olist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}
DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
STAGE_COLUMNS = {
    "До подтверждения оплаты": "payment_approval_days",
    "Подготовка и передача перевозчику": "seller_dispatch_days",
    "Перевозка до покупателя": "carrier_transit_days",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Считает метрики доставки и формирует таблицы и графики."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BASE_DIR / "data" / "raw",
        help="Папка с исходными CSV-файлами Olist (по умолчанию: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "reports",
        help="Папка для результатов (по умолчанию: reports).",
    )
    parser.add_argument(
        "--min-state-orders",
        type=int,
        default=100,
        help="Минимальное число заказов в штате для регионального среза.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def read_source(data_dir: Path, key: str) -> pd.DataFrame:
    path = data_dir / RAW_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}. Скачайте публичный датасет Olist и "
            "положите нужные CSV в data/raw. Инструкция: data/README.md"
        )
    return pd.read_csv(path)


def duration_days(start: pd.Series, end: pd.Series) -> pd.Series:
    """Разница в днях; отрицательные интервалы считаются некорректными."""
    result = (end - start).dt.total_seconds() / 86_400
    return result.where(result >= 0)


def prepare_orders(data_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    orders = read_source(data_dir, "orders")
    required_order_columns = {
        "order_id",
        "customer_id",
        "order_status",
        *DATE_COLUMNS,
    }
    missing = sorted(required_order_columns.difference(orders.columns))
    if missing:
        raise ValueError(f"В таблице заказов не хватает колонок: {', '.join(missing)}")

    duplicate_order_rows = int(orders.duplicated("order_id").sum())
    orders = orders.drop_duplicates("order_id", keep="first").copy()
    orders[DATE_COLUMNS] = orders[DATE_COLUMNS].apply(
        pd.to_datetime, errors="coerce"
    )

    delivered = orders.loc[orders["order_status"].eq("delivered")].copy()
    total_delivered = len(delivered)
    has_dates = delivered[
        [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].notna().all(axis=1)
    valid_ordering = (
        delivered["order_delivered_customer_date"]
        >= delivered["order_purchase_timestamp"]
    ) & (
        delivered["order_estimated_delivery_date"]
        >= delivered["order_purchase_timestamp"]
    )
    valid = has_dates & valid_ordering
    orders_without_core_dates = int((~has_dates).sum())
    invalid_ordering = int((has_dates & ~valid_ordering).sum())
    delivered = delivered.loc[valid].copy()

    delivered["actual_delivery_days"] = duration_days(
        delivered["order_purchase_timestamp"],
        delivered["order_delivered_customer_date"],
    )
    delivered["promised_delivery_days"] = duration_days(
        delivered["order_purchase_timestamp"],
        delivered["order_estimated_delivery_date"],
    )
    delivered["delay_days"] = (
        delivered["order_delivered_customer_date"]
        - delivered["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86_400
    delivered["delivery_status_ru"] = np.where(
        delivered["delay_days"] > 0, "С опозданием", "В срок"
    )
    delivered["purchase_month"] = (
        delivered["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )

    delivered["payment_approval_days"] = duration_days(
        delivered["order_purchase_timestamp"], delivered["order_approved_at"]
    )
    delivered["seller_dispatch_days"] = duration_days(
        delivered["order_approved_at"], delivered["order_delivered_carrier_date"]
    )
    delivered["carrier_transit_days"] = duration_days(
        delivered["order_delivered_carrier_date"],
        delivered["order_delivered_customer_date"],
    )

    reviews = read_source(data_dir, "reviews")
    if not {"order_id", "review_score"}.issubset(reviews.columns):
        raise ValueError("В таблице отзывов должны быть колонки order_id и review_score")
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    reviews = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("review_score", "count"),
    )
    delivered = delivered.merge(reviews, on="order_id", how="left")

    customers = read_source(data_dir, "customers")
    if not {"customer_id", "customer_state"}.issubset(customers.columns):
        raise ValueError(
            "В таблице покупателей должны быть колонки customer_id и customer_state"
        )
    customers = customers[["customer_id", "customer_state"]].drop_duplicates(
        "customer_id", keep="first"
    )
    delivered = delivered.merge(customers, on="customer_id", how="left")

    counts = {
        "order_rows_removed_as_duplicates": duplicate_order_rows,
        "delivered_orders_in_source": int(total_delivered),
        "delivered_orders_missing_core_dates": orders_without_core_dates,
        "delivered_orders_with_invalid_date_order": invalid_ordering,
        "delivered_orders_in_analysis": int(len(delivered)),
    }
    return delivered, counts


def make_tables(
    orders: pd.DataFrame, min_state_orders: int
) -> dict[str, pd.DataFrame]:
    late = orders["delivery_status_ru"].eq("С опозданием")

    overall = pd.DataFrame(
        [
            {
                "Заказов в анализе": len(orders),
                "Доля опозданий, %": late.mean() * 100,
                "Медиана фактической доставки, дней": orders[
                    "actual_delivery_days"
                ].median(),
                "Медиана срока, обещанного покупателю, дней": orders[
                    "promised_delivery_days"
                ].median(),
                "Медиана отклонения от обещанного срока, дней": orders[
                    "delay_days"
                ].median(),
            }
        ]
    )

    state = (
        orders.dropna(subset=["customer_state"])
        .groupby("customer_state", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_orders=("delay_days", lambda values: int((values > 0).sum())),
            late_rate=("delay_days", lambda values: (values > 0).mean() * 100),
            median_delay_days=("delay_days", "median"),
        )
    )
    state = state.loc[state["orders"] >= min_state_orders].sort_values(
        "late_rate", ascending=False
    )

    monthly = (
        orders.groupby("purchase_month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_orders=("delay_days", lambda values: int((values > 0).sum())),
            late_rate=("delay_days", lambda values: (values > 0).mean() * 100),
            median_delivery_days=("actual_delivery_days", "median"),
            median_delay_days=("delay_days", "median"),
        )
        .sort_values("purchase_month")
    )

    stage = (
        orders.groupby("delivery_status_ru", as_index=False)
        .agg(
            payment_approval_days=("payment_approval_days", "median"),
            seller_dispatch_days=("seller_dispatch_days", "median"),
            carrier_transit_days=("carrier_transit_days", "median"),
            orders=("order_id", "nunique"),
        )
    )
    review = (
        orders.groupby("delivery_status_ru", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            reviewed_orders=("review_score", "count"),
            average_review_score=("review_score", "mean"),
            one_star_rate=("review_score", lambda values: values.eq(1).mean() * 100),
        )
    )

    return {
        "overall_metrics": overall,
        "late_rate_by_customer_state": state,
        "monthly_delivery_metrics": monthly,
        "stage_duration_by_delivery_status": stage,
        "reviews_by_delivery_status": review,
    }


def save_figures(tables: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial"]

    stage = tables["stage_duration_by_delivery_status"].set_index(
        "delivery_status_ru"
    )
    stage_values = stage[[*STAGE_COLUMNS.values()]].T
    stage_values.index = list(STAGE_COLUMNS.keys())
    ax = stage_values.plot(kind="bar", figsize=(10, 6), color=["#4C78A8", "#F58518"])
    ax.set_title("Сколько времени занимает каждый этап доставки")
    ax.set_xlabel("")
    ax.set_ylabel("Медиана, дней")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Статус заказа")
    plt.tight_layout()
    plt.savefig(figures_dir / "delivery_stages.png", dpi=160)
    plt.close()

    monthly = tables["monthly_delivery_metrics"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["purchase_month"], monthly["late_rate"], marker="o")
    ax.set_title("Доля заказов с опозданием по месяцу покупки")
    ax.set_xlabel("Месяц покупки")
    ax.set_ylabel("Опоздания, %")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "late_rate_by_month.png", dpi=160)
    plt.close(fig)

    review = tables["reviews_by_delivery_status"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        review["delivery_status_ru"],
        review["average_review_score"],
        color=["#4C78A8", "#F58518"],
    )
    ax.set_title("Средняя оценка заказа по соблюдению срока")
    ax.set_xlabel("")
    ax.set_ylabel("Средняя оценка, из 5")
    ax.set_ylim(0, 5)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(figures_dir / "review_score_by_delivery_status.png", dpi=160)
    plt.close(fig)

    state = tables["late_rate_by_customer_state"].head(15).sort_values("late_rate")
    if not state.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(state["customer_state"], state["late_rate"], color="#72B7B2")
        ax.set_title("Штаты с наибольшей долей опозданий")
        ax.set_xlabel("Заказы с опозданием, %")
        ax.set_ylabel("Штат покупателя")
        fig.tight_layout()
        fig.savefig(figures_dir / "late_rate_by_state.png", dpi=160)
        plt.close(fig)


def json_safe(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    output_dir = resolve_path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    orders, counts = prepare_orders(data_dir)
    if orders.empty:
        raise ValueError("После очистки не осталось доставленных заказов для анализа.")
    tables = make_tables(orders, args.min_state_orders)

    for table_name, table in tables.items():
        table.to_csv(tables_dir / f"{table_name}.csv", index=False, encoding="utf-8-sig")

    orders.to_csv(
        tables_dir / "order_delivery_metrics.csv", index=False, encoding="utf-8-sig"
    )
    save_figures(tables, figures_dir)

    summary = {
        "data_source": "Brazilian E-Commerce Public Dataset by Olist",
        "metric_definition": "late = actual delivery date is later than estimated delivery date",
        "quality_counts": counts,
        "overall_metrics": {
            key: json_safe(value)
            for key, value in tables["overall_metrics"].iloc[0].to_dict().items()
        },
        "notes": [
            "География определяется по штату покупателя.",
            "Если по заказу несколько отзывов, для среднего используется средний балл; доля отзывов в таблице заказов не размножается на число товарных строк.",
            "Этапные интервалы с пропущенными или обратными датами остаются пустыми и не попадают в медианы этапов.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Готово. Заказов в анализе: {len(orders):,}".replace(",", " "))
    print(f"Таблицы: {tables_dir}")
    print(f"Графики: {figures_dir}")


if __name__ == "__main__":
    main()
