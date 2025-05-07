use super::*;

#[test]
fn test_format_with_underscores() {
    assert_eq!("0", format_with_underscores(0));
    assert_eq!("1", format_with_underscores(1));
    assert_eq!("99", format_with_underscores(99));
    assert_eq!("345", format_with_underscores(345));
    assert_eq!("1_000", format_with_underscores(1_000));
    assert_eq!("12_001", format_with_underscores(12_001));
    assert_eq!("999_999", format_with_underscores(999_999));
    assert_eq!("1_000_000", format_with_underscores(1_000_000));
    assert_eq!("12_345_678", format_with_underscores(12_345_678));
}
