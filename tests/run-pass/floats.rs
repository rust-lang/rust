
fn main() {
    assert_eq!(6.0_f32*6.0_f32, 36.0_f32);
    assert_eq!(6.0_f64*6.0_f64, 36.0_f64);
    assert_eq!(-{5.0_f32}, -5.0_f32);
    assert!((5.0_f32/0.0).is_infinite());
    assert!((-5.0_f32).sqrt().is_nan());
}
