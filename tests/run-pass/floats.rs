fn main() {
    assert_eq!(6.0_f32*6.0_f32, 36.0_f32);
    assert_eq!(6.0_f64*6.0_f64, 36.0_f64);
    assert_eq!(-{5.0_f32}, -5.0_f32);
    assert!((5.0_f32/0.0).is_infinite());
    assert!((-5.0_f32).sqrt().is_nan());
    let x: u64 = unsafe { std::mem::transmute(42.0_f64) };
    let y: f64 = unsafe { std::mem::transmute(x) };
    assert_eq!(y, 42.0_f64);

    assert_eq!(5.0f32 as u32, 5);
    assert_eq!(5.0f32 as i32, 5);
    assert_eq!(-5.0f32 as i32, -5);

    assert_eq!((1.0 as f32).max(-1.0), 1.0);
    assert_eq!((1.0 as f32).min(-1.0), -1.0);
    assert_eq!(std::f32::NAN.min(9.0), 9.0);
    assert_eq!(std::f32::NAN.max(-9.0), -9.0);
    assert_eq!((9.0 as f32).min(std::f32::NAN), 9.0);
    assert_eq!((-9.0 as f32).max(std::f32::NAN), -9.0);

    assert_eq!((1.0 as f64).max(-1.0), 1.0);
    assert_eq!((1.0 as f64).min(-1.0), -1.0);
    assert_eq!(std::f64::NAN.min(9.0), 9.0);
    assert_eq!(std::f64::NAN.max(-9.0), -9.0);
    assert_eq!((9.0 as f64).min(std::f64::NAN), 9.0);
    assert_eq!((-9.0 as f64).max(std::f64::NAN), -9.0);
}
