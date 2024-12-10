#[test]
fn test_normal_form() {
    use crate::f32::div_euclid::normal_form;

    assert_eq!(normal_form(-1.5f32), Some((true, -23, 0xc00000)));
    assert_eq!(normal_form(f32::MIN_POSITIVE), Some((false, -149, 0x800000)));
    assert_eq!(normal_form(f32::MIN_POSITIVE / 2.0), Some((false, -150, 0x800000)));
    assert_eq!(normal_form(f32::MAX), Some((false, 104, 0xffffff)));
    assert_eq!(normal_form(0.0), None);
    assert_eq!(normal_form(f32::INFINITY), None);
    assert_eq!(normal_form(f32::NAN), None);
}

#[test]
fn test_pow2() {
    use crate::f32::div_euclid::pow2;

    assert_eq!(pow2(0), 1.0f32);
    assert_eq!(pow2(4), 16.0f32);
    assert_eq!(pow2(128), f32::INFINITY);
}
