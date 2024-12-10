#[cfg(reliable_f16_math)]
#[test]
fn test_normal_form() {
    use crate::f16::div_euclid::normal_form;

    assert_eq!(normal_form(-1.5f16), Some((true, -10, 0x600)));
    assert_eq!(normal_form(f16::MIN_POSITIVE), Some((false, -24, 0x400)));
    assert_eq!(normal_form(f16::MIN_POSITIVE / 2.0), Some((false, -25, 0x400)));
    assert_eq!(normal_form(f16::MAX), Some((false, 5, 0x7ff)));
    assert_eq!(normal_form(0.0), None);
    assert_eq!(normal_form(f16::INFINITY), None);
    assert_eq!(normal_form(f16::NAN), None);
}
