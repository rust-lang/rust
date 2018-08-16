
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
}
