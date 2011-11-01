
use std;
import std::int;
import std::str::eq;
import std::str::bytes;

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
#[should_fail]
fn test_from_str_fail_1() {
    int::from_str(" ");
}

#[test]
#[should_fail]
fn test_from_str_fail_2() {
    int::from_str("x");
}

#[test]
fn test_parse_buf() {
    assert (int::parse_buf(bytes("123"), 10u) == 123);
    assert (int::parse_buf(bytes("1001"), 2u) == 9);
    assert (int::parse_buf(bytes("123"), 8u) == 83);
    assert (int::parse_buf(bytes("123"), 16u) == 291);
    assert (int::parse_buf(bytes("ffff"), 16u) == 65535);
    assert (int::parse_buf(bytes("FFFF"), 16u) == 65535);
    assert (int::parse_buf(bytes("z"), 36u) == 35);
    assert (int::parse_buf(bytes("Z"), 36u) == 35);

    assert (int::parse_buf(bytes("-123"), 10u) == -123);
    assert (int::parse_buf(bytes("-1001"), 2u) == -9);
    assert (int::parse_buf(bytes("-123"), 8u) == -83);
    assert (int::parse_buf(bytes("-123"), 16u) == -291);
    assert (int::parse_buf(bytes("-ffff"), 16u) == -65535);
    assert (int::parse_buf(bytes("-FFFF"), 16u) == -65535);
    assert (int::parse_buf(bytes("-z"), 36u) == -35);
    assert (int::parse_buf(bytes("-Z"), 36u) == -35);
}

#[test]
#[should_fail]
fn test_parse_buf_fail_1() {
    int::parse_buf(bytes("Z"), 35u);
}

#[test]
#[should_fail]
fn test_parse_buf_fail_2() {
    int::parse_buf(bytes("-9"), 2u);
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
