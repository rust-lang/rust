use super::*;

#[test]
fn int_format_decimal() {
    assert_eq!(format_integer_with_underscore_sep(12345678, false), "12_345_678");
    assert_eq!(format_integer_with_underscore_sep(123, false), "123");
    assert_eq!(format_integer_with_underscore_sep(123459, false), "123_459");
    assert_eq!(format_integer_with_underscore_sep(12345678, true), "-12_345_678");
    assert_eq!(format_integer_with_underscore_sep(123, true), "-123");
    assert_eq!(format_integer_with_underscore_sep(123459, true), "-123_459");
}
