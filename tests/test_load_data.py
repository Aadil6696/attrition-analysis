import pandas as pd
import pytest

from src.load_data import clean_employee_data, load_employee_data


def _row(**overrides):
    base = {
        "employee_id": 1,
        "department": "Sales",
        "age": 30,
        "monthly_income": 5000,
        "job_satisfaction": 3,
        "overtime": "No",
        "travel_frequency": "Rarely",
        "years_at_company": 2,
        "attrition": "No",
    }
    base.update(overrides)
    return base


# ---------- clean_employee_data ----------

def test_clean_raises_on_missing_column():
    df = pd.DataFrame({"employee_id": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_strips_whitespace_in_department():
    df = pd.DataFrame([_row(department="  Sales  ")])
    result = clean_employee_data(df)
    assert result.loc[0, "department"] == "Sales"


def test_clean_normalizes_attrition_casing_and_whitespace():
    df = pd.DataFrame(
        [
            _row(employee_id=1, attrition="yes"),
            _row(employee_id=2, attrition="NO "),
            _row(employee_id=3, attrition=" Yes"),
        ]
    )
    result = clean_employee_data(df)
    assert list(result["attrition"]) == ["Yes", "No", "Yes"]


def test_clean_fills_missing_department_with_unknown():
    df = pd.DataFrame([_row(department=None)])
    result = clean_employee_data(df)
    assert result.loc[0, "department"] == "Unknown"


def test_clean_fills_missing_overtime_with_no():
    df = pd.DataFrame([_row(overtime=None)])
    result = clean_employee_data(df)
    assert result.loc[0, "overtime"] == "No"


def test_clean_fills_missing_travel_with_rarely():
    df = pd.DataFrame([_row(travel_frequency=None)])
    result = clean_employee_data(df)
    assert result.loc[0, "travel_frequency"] == "Rarely"


def test_clean_fills_missing_satisfaction_with_three():
    df = pd.DataFrame([_row(job_satisfaction=None)])
    result = clean_employee_data(df)
    assert result.loc[0, "job_satisfaction"] == 3


def test_clean_fills_missing_income_with_median():
    df = pd.DataFrame(
        [
            _row(employee_id=1, monthly_income=1000),
            _row(employee_id=2, monthly_income=3000),
            _row(employee_id=3, monthly_income=None),
        ]
    )
    result = clean_employee_data(df)
    # Median of [1000, 3000] is 2000
    assert result.loc[2, "monthly_income"] == 2000


def test_clean_does_not_mutate_input():
    df = pd.DataFrame([_row(department="  Sales  ", attrition="yes")])
    original = df.copy()
    clean_employee_data(df)
    pd.testing.assert_frame_equal(df, original)


# ---------- load_employee_data ----------

def test_load_employee_data_reads_csv(tmp_path):
    csv = tmp_path / "employees.csv"
    csv.write_text(
        "employee_id,department,age,monthly_income,job_satisfaction,"
        "overtime,travel_frequency,years_at_company,attrition\n"
        "1,Sales,30,5000,3,No,Rarely,2,No\n"
        "2,HR,40,7000,4,Yes,Frequent,8,Yes\n"
    )
    df = load_employee_data(str(csv))
    assert len(df) == 2
    assert list(df.columns)[0] == "employee_id"
    assert df.loc[0, "department"] == "Sales"
    assert df.loc[1, "attrition"] == "Yes"
