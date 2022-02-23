"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[2, 5], [4, 7], [6, 9]], [2, 5]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
    ])
def test_daily_min_integers(test, expected):
    """Test that min function works for an array of integers."""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[2, 5], [4, 7], [6, 9]], [6, 9]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
    ])
def test_daily_max_integers(test, expected):
    """Test that min function works for an array of integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected, raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([['a', 'd'], ['b', 'e'], ['c', 'f']], None, TypeError),
        (10, None, ValueError),
        ([[-1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], ValueError),
        ([[np.nan, 0, 0], [0, np.nan, 0], [0, 0, np.nan]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[np.inf, 0, 0], [0, np.inf, 0], [0, 0, np.inf]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None),
    ])
def test_patient_normalise(test, expected, raises):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    if raises:
        with pytest.raises(raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [1.632993161855452, 1.632993161855452]),
    ])
def test_daily_std(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_std
    npt.assert_array_equal(daily_std(np.array(test)), np.array(expected))