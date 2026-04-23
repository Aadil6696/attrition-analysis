import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Small employee dataset with known, hand-verifiable attrition patterns.

    Summary:
      6 employees, 3 leavers (overall attrition = 50%).
      Departments: Sales (2 emp, 1 leaver), HR (3 emp, 1 leaver), IT (1 emp, 1 leaver).
      Overtime=Yes leavers: 3/3 (100%). Overtime=No leavers: 0/3 (0%).
      Avg income Yes=4500.00, No=6166.67.
      Satisfaction rates: 1->100%, 2->100%, 3->0%, 4->0%.
    """
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "HR", "IT"],
            "age": [30, 35, 28, 40, 32, 26],
            "monthly_income": [5000, 6000, 4000, 7000, 5500, 4500],
            "job_satisfaction": [2, 4, 1, 4, 3, 2],
            "overtime": ["Yes", "No", "Yes", "No", "No", "Yes"],
            "travel_frequency": [
                "Frequent",
                "Rarely",
                "Frequent",
                "Rarely",
                "Occasional",
                "Frequent",
            ],
            "years_at_company": [2, 5, 1, 10, 4, 2],
            "attrition": ["Yes", "No", "Yes", "No", "No", "Yes"],
        }
    )
