import pandas as pd

from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# ---------- attrition_rate ----------

def test_attrition_rate_basic(sample_df):
    assert attrition_rate(sample_df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_is_rounded_to_two_decimals():
    df = pd.DataFrame({"employee_id": [1, 2, 3], "attrition": ["Yes", "No", "No"]})
    assert attrition_rate(df) == 33.33


# ---------- attrition_by_department ----------

def test_attrition_by_department_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(sample_df):
    result = attrition_by_department(sample_df).set_index("department")
    assert result.loc["IT", "attrition_rate"] == 100.0
    assert result.loc["Sales", "attrition_rate"] == 50.0
    assert result.loc["HR", "attrition_rate"] == 33.33
    assert result.loc["HR", "employees"] == 3
    assert result.loc["HR", "leavers"] == 1


def test_attrition_by_department_sorted_descending_by_rate(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result["department"]) == ["IT", "Sales", "HR"]


# ---------- attrition_by_overtime ----------

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(sample_df):
    result = attrition_by_overtime(sample_df).set_index("overtime")
    assert result.loc["Yes", "attrition_rate"] == 100.0
    assert result.loc["No", "attrition_rate"] == 0.0
    assert result.loc["Yes", "employees"] == 3
    assert result.loc["No", "employees"] == 3


# ---------- average_income_by_attrition ----------

def test_average_income_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_values(sample_df):
    result = average_income_by_attrition(sample_df).set_index("attrition")
    # Leavers: 5000, 4000, 4500 -> 4500.00
    # Stayers: 6000, 7000, 5500 -> 6166.67
    assert result.loc["Yes", "avg_monthly_income"] == 4500.00
    assert result.loc["No", "avg_monthly_income"] == 6166.67


# ---------- satisfaction_summary ----------

def test_satisfaction_summary_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == [
        "job_satisfaction",
        "total_employees",
        "leavers",
        "attrition_rate",
    ]


def test_satisfaction_summary_values(sample_df):
    result = satisfaction_summary(sample_df).set_index("job_satisfaction")
    assert result.loc[1, "attrition_rate"] == 100.0
    assert result.loc[2, "attrition_rate"] == 100.0
    assert result.loc[3, "attrition_rate"] == 0.0
    assert result.loc[4, "attrition_rate"] == 0.0
    assert result.loc[2, "total_employees"] == 2
    assert result.loc[4, "total_employees"] == 2


def test_satisfaction_summary_sorted_ascending_by_score(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result["job_satisfaction"]) == [1, 2, 3, 4]


def test_satisfaction_summary_denominator_is_group_size():
    # Regression guard: rate must be leavers-in-group / employees-in-group,
    # not leavers-in-group / total-leavers-in-dataset.
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5],
            "job_satisfaction": [1, 1, 2, 2, 2],
            "attrition": ["Yes", "No", "Yes", "No", "No"],
        }
    )
    result = satisfaction_summary(df).set_index("job_satisfaction")
    # Group 1: 1 leaver / 2 employees -> 50%
    # Group 2: 1 leaver / 3 employees -> 33.33%
    # Old buggy formula (leavers / 2 total leavers) would yield 50% for both.
    assert result.loc[1, "attrition_rate"] == 50.0
    assert result.loc[2, "attrition_rate"] == 33.33
