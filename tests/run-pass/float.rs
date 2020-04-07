#![feature(track_caller)]
use std::fmt::Debug;

// Helper function to avoid promotion so that this tests "run-time" casts, not CTFE.
// Doesn't make a big difference when running this in Miri, but it means we can compare this
// with the LLVM backend by running `rustc -Zmir-opt-level=0 -Zsaturating-float-casts`.
#[track_caller]
#[inline(never)]
fn assert_eq<T: PartialEq + Debug>(x: T, y: T) {
    assert_eq!(x, y);
}

fn main() {
    // basic arithmetic
    assert_eq(6.0_f32*6.0_f32, 36.0_f32);
    assert_eq(6.0_f64*6.0_f64, 36.0_f64);
    assert_eq(-{5.0_f32}, -5.0_f32);
    assert_eq(-{5.0_f64}, -5.0_f64);
    // infinities, NaN
    assert!((5.0_f32/0.0).is_infinite());
    assert!((5.0_f64/0.0).is_infinite());
    assert!((-5.0_f32).sqrt().is_nan());
    assert!((-5.0_f64).sqrt().is_nan());
    // byte-level transmute
    let x: u64 = unsafe { std::mem::transmute(42.0_f64) };
    let y: f64 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f64);
    let x: u32 = unsafe { std::mem::transmute(42.0_f32) };
    let y: f32 = unsafe { std::mem::transmute(x) };
    assert_eq(y, 42.0_f32);

    // f32 <-> int casts
    assert_eq(5.0f32 as u32, 5);
    assert_eq(-5.0f32 as u32, 0);
    assert_eq(5.0f32 as i32, 5);
    assert_eq(-5.0f32 as i32, -5);
    assert_eq(f32::MAX as i32, i32::MAX);
    assert_eq(f32::INFINITY as i32, i32::MAX);
    assert_eq(f32::MAX as u32, u32::MAX);
    assert_eq(f32::INFINITY as u32, u32::MAX);
    assert_eq(f32::MIN as i32, i32::MIN);
    assert_eq(f32::NEG_INFINITY as i32, i32::MIN);
    assert_eq(f32::MIN as u32, 0);
    assert_eq(f32::NEG_INFINITY as u32, 0);
    assert_eq(f32::NAN as i32, 0);
    assert_eq(f32::NAN as u32, 0);
    assert_eq((u32::MAX-127) as f32 as u32, u32::MAX); // rounding loss
    assert_eq((u32::MAX-128) as f32 as u32, u32::MAX-255); // rounding loss
    assert_eq(127i8 as f32, 127.0f32);
    assert_eq(i128::MIN as f32, -170141183460469231731687303715884105728.0f32);
    assert_eq(u128::MAX as f32, f32::INFINITY); // saturation

    // f64 <-> int casts
    assert_eq(5.0f64 as u64, 5);
    assert_eq(-5.0f64 as u64, 0);
    assert_eq(5.0f64 as i64, 5);
    assert_eq(-5.0f64 as i64, -5);
    assert_eq(f64::MAX as i64, i64::MAX);
    assert_eq(f64::INFINITY as i64, i64::MAX);
    assert_eq(f64::MAX as u64, u64::MAX);
    assert_eq(f64::INFINITY as u64, u64::MAX);
    assert_eq(f64::MIN as i64, i64::MIN);
    assert_eq(f64::NEG_INFINITY as i64, i64::MIN);
    assert_eq(f64::MIN as u64, 0);
    assert_eq(f64::NEG_INFINITY as u64, 0);
    assert_eq(f64::NAN as i64, 0);
    assert_eq(f64::NAN as u64, 0);
    assert_eq((u64::MAX-1023) as f64 as u64, u64::MAX); // rounding loss
    assert_eq((u64::MAX-1024) as f64 as u64, u64::MAX-2047); // rounding loss
    assert_eq(u128::MAX as f64 as u128, u128::MAX);
    assert_eq(i16::MIN as f64, -32768.0f64);
    assert_eq(u128::MAX as f64, 340282366920938463463374607431768211455.0f64); // even that fits...

    // f32 <-> f64 casts
    assert_eq(5.0f64 as f32, 5.0f32);
    assert_eq(5.0f32 as f64, 5.0f64);
    assert_eq(f64::MAX as f32, f32::INFINITY);
    assert_eq(f64::MIN as f32, f32::NEG_INFINITY);
    assert_eq(f32::INFINITY as f64, f64::INFINITY);
    assert_eq(f32::NEG_INFINITY as f64, f64::NEG_INFINITY);

    // f32 min/max
    assert_eq((1.0 as f32).max(-1.0), 1.0);
    assert_eq((1.0 as f32).min(-1.0), -1.0);
    assert_eq(f32::NAN.min(9.0), 9.0);
    assert_eq(f32::NAN.max(-9.0), -9.0);
    assert_eq((9.0 as f32).min(f32::NAN), 9.0);
    assert_eq((-9.0 as f32).max(f32::NAN), -9.0);

    // f64 min/max
    assert_eq((1.0 as f64).max(-1.0), 1.0);
    assert_eq((1.0 as f64).min(-1.0), -1.0);
    assert_eq(f64::NAN.min(9.0), 9.0);
    assert_eq(f64::NAN.max(-9.0), -9.0);
    assert_eq((9.0 as f64).min(f64::NAN), 9.0);
    assert_eq((-9.0 as f64).max(f64::NAN), -9.0);

    // f32 copysign
    assert_eq(3.5_f32.copysign(0.42), 3.5_f32);
    assert_eq(3.5_f32.copysign(-0.42), -3.5_f32);
    assert_eq((-3.5_f32).copysign(0.42), 3.5_f32);
    assert_eq((-3.5_f32).copysign(-0.42), -3.5_f32);
    assert!(f32::NAN.copysign(1.0).is_nan());

    // f64 copysign
    assert_eq(3.5_f64.copysign(0.42), 3.5_f64);
    assert_eq(3.5_f64.copysign(-0.42), -3.5_f64);
    assert_eq((-3.5_f64).copysign(0.42), 3.5_f64);
    assert_eq((-3.5_f64).copysign(-0.42), -3.5_f64);
    assert!(f64::NAN.copysign(1.0).is_nan());
}
