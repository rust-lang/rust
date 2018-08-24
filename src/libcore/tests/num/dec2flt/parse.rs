use core::num::dec2flt::parse::{Decimal, parse_decimal};
use core::num::dec2flt::parse::ParseResult::{Valid, Invalid};

#[test]
fn missing_pieces() {
    let permutations = &[".e", "1e", "e4", "e", ".12e", "321.e", "32.12e+", "12.32e-"];
    for &s in permutations {
        assert_eq!(parse_decimal(s), Invalid);
    }
}

#[test]
fn invalid_chars() {
    let invalid = "r,?<j";
    let valid_strings = &["123", "666.", ".1", "5e1", "7e-3", "0.0e+1"];
    for c in invalid.chars() {
        for s in valid_strings {
            for i in 0..s.len() {
                let mut input = String::new();
                input.push_str(s);
                input.insert(i, c);
                assert!(parse_decimal(&input) == Invalid, "did not reject invalid {:?}", input);
            }
        }
    }
}

#[test]
fn valid() {
    assert_eq!(parse_decimal("123.456e789"), Valid(Decimal::new(b"123", b"456", 789)));
    assert_eq!(parse_decimal("123.456e+789"), Valid(Decimal::new(b"123", b"456", 789)));
    assert_eq!(parse_decimal("123.456e-789"), Valid(Decimal::new(b"123", b"456", -789)));
    assert_eq!(parse_decimal(".050"), Valid(Decimal::new(b"", b"050", 0)));
    assert_eq!(parse_decimal("999"), Valid(Decimal::new(b"999", b"", 0)));
    assert_eq!(parse_decimal("1.e300"), Valid(Decimal::new(b"1", b"", 300)));
    assert_eq!(parse_decimal(".1e300"), Valid(Decimal::new(b"", b"1", 300)));
    assert_eq!(parse_decimal("101e-33"), Valid(Decimal::new(b"101", b"", -33)));
    let zeros = "0".repeat(25);
    let s = format!("1.5e{}", zeros);
    assert_eq!(parse_decimal(&s), Valid(Decimal::new(b"1", b"5", 0)));
}
