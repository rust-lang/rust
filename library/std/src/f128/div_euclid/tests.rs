#![cfg(reliable_f128_math)]

#[test]
fn test_normal_form() {
    use crate::f128::div_euclid::normal_form;

    assert_eq!(normal_form(-1.5f128), Some((true, -112, 3 << 111)));
    assert_eq!(normal_form(f128::MIN_POSITIVE), Some((false, -16494, 1 << 112)));
    assert_eq!(normal_form(f128::MIN_POSITIVE / 2.0), Some((false, -16495, 1 << 112)));
    assert_eq!(normal_form(f128::MAX), Some((false, 16271, (1 << 113) - 1)));
    assert_eq!(normal_form(0.0), None);
    assert_eq!(normal_form(f128::INFINITY), None);
    assert_eq!(normal_form(f128::NAN), None);
}

#[test]
fn test_pow2() {
    use crate::f128::div_euclid::pow2;

    assert_eq!(pow2(0), 1.0f128);
    assert_eq!(pow2(4), 16.0f128);
    assert_eq!(pow2(16384), f128::INFINITY);
}
