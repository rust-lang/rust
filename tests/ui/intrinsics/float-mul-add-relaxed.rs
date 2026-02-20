//@ run-pass
//@ compile-flags: -O

#![feature(float_mul_add_relaxed)]

fn main() {
    test_f32();
    test_f64();
}

fn test_f32() {
    let a = 2.0_f32;
    let b = 3.0_f32;
    let c = 4.0_f32;

    let result = a.mul_add_relaxed(b, c);
    assert_eq!(result, 10.0);

    // Test with values where precision matters less
    let x = 1.0_f32;
    let y = 1.0_f32;
    let z = 1.0_f32;
    assert_eq!(x.mul_add_relaxed(y, z), 2.0);

    // Test edge cases
    assert!(f32::NAN.mul_add_relaxed(1.0, 1.0).is_nan());
    assert_eq!(f32::INFINITY.mul_add_relaxed(2.0, 1.0), f32::INFINITY);
    assert!(0.0_f32.mul_add_relaxed(f32::INFINITY, 1.0).is_nan());
}

fn test_f64() {
    let a = 2.0_f64;
    let b = 3.0_f64;
    let c = 4.0_f64;

    let result = a.mul_add_relaxed(b, c);
    assert_eq!(result, 10.0);

    // Test with values where precision matters less
    let x = 1.0_f64;
    let y = 1.0_f64;
    let z = 1.0_f64;
    assert_eq!(x.mul_add_relaxed(y, z), 2.0);

    // Test edge cases
    assert!(f64::NAN.mul_add_relaxed(1.0, 1.0).is_nan());
    assert_eq!(f64::INFINITY.mul_add_relaxed(2.0, 1.0), f64::INFINITY);
    assert!(0.0_f64.mul_add_relaxed(f64::INFINITY, 1.0).is_nan());
}
