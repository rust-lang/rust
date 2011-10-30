
use std;
import std::int;
import std::str::eq;

#[test]
fn test_from_str() {
    assert(int::from_str("0") == 0);
    assert(int::from_str("3") == 3);
    assert(int::from_str("10") == 10);
    assert(int::from_str("123456789") == 123456789);
    assert(int::from_str("00100") == 100);

    assert(int::from_str("-1") == -1);
    assert(int::from_str("-3") == -3);
    assert(int::from_str("-10") == -10);
    assert(int::from_str("-123456789") == -123456789);
    assert(int::from_str("-00100") == -100);
}

#[test]
fn test_to_str() {
    assert (eq(int::to_str(0, 10u), "0"));
    assert (eq(int::to_str(1, 10u), "1"));
    assert (eq(int::to_str(-1, 10u), "-1"));
    assert (eq(int::to_str(255, 16u), "ff"));
    assert (eq(int::to_str(100, 10u), "100"));
}

#[test]
fn test_pow() {
    assert (int::pow(0, 0u) == 1);
    assert (int::pow(0, 1u) == 0);
    assert (int::pow(0, 2u) == 0);
    assert (int::pow(-1, 0u) == 1);
    assert (int::pow(1, 0u) == 1);
    assert (int::pow(-3, 2u) == 9);
    assert (int::pow(-3, 3u) == -27);
    assert (int::pow(4, 9u) == 262144);
}

#[test]
fn test_overflows() {
   assert (int::max_value() > 0);
   assert (int::min_value() <= 0);
   assert (int::min_value() + int::max_value() + 1 == 0);
}
