"""
    Test Data Manipulation
"""

# Python Dependencies
import pytest
import os

from DoseResponse.dose_response_curve import DoseResponseCurve
from DoseResponse.equations import Equations

# Global Variables
PARENT_DIRECTORY = os.path.dirname(os.path.dirname(__file__))
TEST_DATA_PATH = os.path.join(PARENT_DIRECTORY, 'SampleData', '2comp_test.txt')


def test_data_load_and_summary():
    x = DoseResponseCurve(
        datafile=TEST_DATA_PATH,
        method=Equations.VariableSlopeDoseResponse
    )

    # Testing df main Data Loading
    assert x.df_main is None
    x.data_summary()
    assert x.df_main is not None
    assert x.df_main.loc[('Compound A', '666.6666667'), 0] == 108.0
    assert x.df_main.loc[('Compound B', '8.230452675'), 1] == 51.0
    assert len(x.df_main.columns) == 3

    # Testing Summarization of Data Works as Intended
    assert x.df_summary is not None
    assert x.df_summary.loc[('Compound B', '222.2222222'), 'MEAN'] == 73.0
    assert x.df_summary.loc[('Compound A', '74.07407407'), 'N'] == 2


def test_scatterplot():
    x = DoseResponseCurve(
        datafile=TEST_DATA_PATH,
        method=Equations.VariableSlopeDoseResponse
    )
    x.data_summary()
    assert x.plot is None
    x.scatterplot()
    assert x.plot is not None
    assert x.df_plot_ready is not None

    # Test Fitting Parameters
    assert x.fit_parameters is not None
    assert 38.0 < x.fit_parameters.loc['Compound A', 'EC50'] < 38.1
    assert 297.0 < x.fit_parameters.loc['Compound B', 'EC50'] < 298.0
    assert 39.8 < x.fit_parameters.loc['Compound C', 'Top'] < 41.0
    assert 1.60 < x.fit_parameters.loc['Compound A', 'HillSlope'] < 1.70
