use core::num::dec2flt::decimal_seq::{DecimalSeq, parse_decimal_seq};

#[test]
fn test_trim() {
    let mut dec = DecimalSeq::default();
    let digits = [1, 2, 3, 4];

    dec.digits[0..4].copy_from_slice(&digits);
    dec.num_digits = 8;
    dec.trim();

    assert_eq!(dec.digits[0..4], digits);
    assert_eq!(dec.num_digits, 4);
}

#[test]
fn test_parse() {
    let tests = [("1.234", [1, 2, 3, 4], 1)];

    for (s, exp_digits, decimal_point) in tests {
        let actual = parse_decimal_seq(s.as_bytes());
        let mut digits = [0; DecimalSeq::MAX_DIGITS];
        digits[..exp_digits.len()].copy_from_slice(&exp_digits);

        let expected =
            DecimalSeq { num_digits: exp_digits.len(), decimal_point, truncated: false, digits };

        assert_eq!(actual, expected);
    }
}
