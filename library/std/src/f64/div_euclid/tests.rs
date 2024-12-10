#[test]
fn test_normal_form() {
    use crate::f64::div_euclid::normal_form;

    assert_eq!(normal_form(-1.5f64), Some((true, -52, 3 << 51)));
    assert_eq!(normal_form(f64::MIN_POSITIVE), Some((false, -1074, 1 << 52)));
    assert_eq!(normal_form(f64::MIN_POSITIVE / 2.0), Some((false, -1075, 1 << 52)));
    assert_eq!(normal_form(f64::MAX), Some((false, 971, (1 << 53) - 1)));
    assert_eq!(normal_form(0.0), None);
    assert_eq!(normal_form(f64::INFINITY), None);
    assert_eq!(normal_form(f64::NAN), None);
}

#[test]
fn test_pow2() {
    use crate::f64::div_euclid::pow2;

    assert_eq!(pow2(0), 1.0f64);
    assert_eq!(pow2(4), 16.0f64);
    assert_eq!(pow2(1024), f64::INFINITY);
}
