use super::*;

#[test]
fn test_to_readable_str() {
    assert_eq!("0", to_readable_str(0));
    assert_eq!("1", to_readable_str(1));
    assert_eq!("99", to_readable_str(99));
    assert_eq!("999", to_readable_str(999));
    assert_eq!("1_000", to_readable_str(1_000));
    assert_eq!("1_001", to_readable_str(1_001));
    assert_eq!("999_999", to_readable_str(999_999));
    assert_eq!("1_000_000", to_readable_str(1_000_000));
    assert_eq!("1_234_567", to_readable_str(1_234_567));
}
