use super::*;

#[test]
fn test_format_with_underscores() {
    assert_eq!("", format_with_underscores("".to_string()));
    assert_eq!("0", format_with_underscores("0".to_string()));
    assert_eq!("12_345.67e14", format_with_underscores("12345.67e14".to_string()));
    assert_eq!("-1_234.5678e10", format_with_underscores("-1234.5678e10".to_string()));
    assert_eq!("------", format_with_underscores("------".to_string()));
    assert_eq!("abcdefgh", format_with_underscores("abcdefgh".to_string()));
    assert_eq!("-1b", format_with_underscores("-1b".to_string()));
    assert_eq!("-3_456xyz", format_with_underscores("-3456xyz".to_string()));
}

#[test]
fn test_usize_with_underscores() {
    assert_eq!("0", usize_with_underscores(0));
    assert_eq!("1", usize_with_underscores(1));
    assert_eq!("99", usize_with_underscores(99));
    assert_eq!("345", usize_with_underscores(345));
    assert_eq!("1_000", usize_with_underscores(1_000));
    assert_eq!("12_001", usize_with_underscores(12_001));
    assert_eq!("999_999", usize_with_underscores(999_999));
    assert_eq!("1_000_000", usize_with_underscores(1_000_000));
    assert_eq!("12_345_678", usize_with_underscores(12_345_678));
}

#[test]
fn test_isize_with_underscores() {
    assert_eq!("0", isize_with_underscores(0));
    assert_eq!("-1", isize_with_underscores(-1));
    assert_eq!("99", isize_with_underscores(99));
    assert_eq!("345", isize_with_underscores(345));
    assert_eq!("-1_000", isize_with_underscores(-1_000));
    assert_eq!("12_001", isize_with_underscores(12_001));
    assert_eq!("-999_999", isize_with_underscores(-999_999));
    assert_eq!("1_000_000", isize_with_underscores(1_000_000));
    assert_eq!("-12_345_678", isize_with_underscores(-12_345_678));
}

#[test]
fn test_f64p1_with_underscores() {
    assert_eq!("0.0", f64p1_with_underscores(0f64));
    assert_eq!("0.0", f64p1_with_underscores(0.00000001));
    assert_eq!("-0.0", f64p1_with_underscores(-0.00000001));
    assert_eq!("1.0", f64p1_with_underscores(0.9999999));
    assert_eq!("-1.0", f64p1_with_underscores(-0.9999999));
    assert_eq!("345.5", f64p1_with_underscores(345.4999999));
    assert_eq!("-100_000.0", f64p1_with_underscores(-100_000f64));
    assert_eq!("123_456_789.1", f64p1_with_underscores(123456789.123456789));
    assert_eq!("-123_456_789.1", f64p1_with_underscores(-123456789.123456789));
}
