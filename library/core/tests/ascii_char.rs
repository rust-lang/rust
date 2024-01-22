use core::ascii::Char;

/// Tests addition of u8 values to ascii::Char;
#[test]
fn test_arithmetic_ok() {
    assert_eq!(Char::Digit8, Char::Digit0 + 8);
    assert_eq!(Char::Colon, Char::Digit0 + 10);
    assert_eq!(Char::Digit8, 8 + Char::Digit0);
    assert_eq!(Char::Colon, 10 + Char::Digit0);

    let mut digit = Char::Digit0;
    digit += 8;
    assert_eq!(Char::Digit8, digit);
}

/// Tests addition wraps values when built in release mode.
#[test]
#[cfg_attr(debug_assertions, ignore = "works in release builds only")]
fn test_arithmetic_wrapping() {
    assert_eq!(Char::Digit0, Char::Digit8 + 120);
    assert_eq!(Char::Digit0, Char::Digit8 + 248);
}

/// Tests addition panics in debug build when it produces an invalid ASCII char.
#[test]
#[cfg_attr(not(debug_assertions), ignore = "works in debug builds only")]
#[should_panic]
fn test_arithmetic_non_ascii() {
    let _ = Char::Digit0 + 120;
}

/// Tests addition panics in debug build when it overflowing u8.
#[test]
#[cfg_attr(not(debug_assertions), ignore = "works in debug builds only")]
#[should_panic]
fn test_arithmetic_overflow() {
    let _ = Char::Digit0 + 250;
}
