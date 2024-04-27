use super::*;

#[test]
fn int_format_decimal() {
    assert_eq!(format_integer_with_underscore_sep("12345678"), "12_345_678");
    assert_eq!(format_integer_with_underscore_sep("123"), "123");
    assert_eq!(format_integer_with_underscore_sep("123459"), "123_459");
    assert_eq!(format_integer_with_underscore_sep("-12345678"), "-12_345_678");
    assert_eq!(format_integer_with_underscore_sep("-123"), "-123");
    assert_eq!(format_integer_with_underscore_sep("-123459"), "-123_459");
}

#[test]
fn int_format_hex() {
    assert_eq!(format_integer_with_underscore_sep("0xab3"), "0xab3");
    assert_eq!(format_integer_with_underscore_sep("0xa2345b"), "0xa2_345b");
    assert_eq!(format_integer_with_underscore_sep("0xa2e6345b"), "0xa2e6_345b");
    assert_eq!(format_integer_with_underscore_sep("-0xab3"), "-0xab3");
    assert_eq!(format_integer_with_underscore_sep("-0xa2345b"), "-0xa2_345b");
    assert_eq!(format_integer_with_underscore_sep("-0xa2e6345b"), "-0xa2e6_345b");
}

#[test]
fn int_format_binary() {
    assert_eq!(format_integer_with_underscore_sep("0o12345671"), "0o12_345_671");
    assert_eq!(format_integer_with_underscore_sep("0o123"), "0o123");
    assert_eq!(format_integer_with_underscore_sep("0o123451"), "0o123451");
    assert_eq!(format_integer_with_underscore_sep("-0o12345671"), "-0o12_345_671");
    assert_eq!(format_integer_with_underscore_sep("-0o123"), "-0o123");
    assert_eq!(format_integer_with_underscore_sep("-0o123451"), "-0o123451");
}

#[test]
fn int_format_octal() {
    assert_eq!(format_integer_with_underscore_sep("0b101"), "0b101");
    assert_eq!(format_integer_with_underscore_sep("0b101101011"), "0b1_0110_1011");
    assert_eq!(format_integer_with_underscore_sep("0b01101011"), "0b0110_1011");
    assert_eq!(format_integer_with_underscore_sep("-0b101"), "-0b101");
    assert_eq!(format_integer_with_underscore_sep("-0b101101011"), "-0b1_0110_1011");
    assert_eq!(format_integer_with_underscore_sep("-0b01101011"), "-0b0110_1011");
}
