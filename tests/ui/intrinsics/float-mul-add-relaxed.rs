//@ run-pass
//@ compile-flags: -O

// Check that `mul_add_relaxed` returns either the fused result (one rounding)
// or the unfused result (two roundings), including with optimizations enabled
// where the operation may be const-folded or lowered to a fused instruction.

#![feature(float_mul_add_relaxed)]
#![feature(f16)]
#![feature(f128)]
#![feature(cfg_target_has_reliable_f16_f128)]
// `f16`/`f128` go unused on targets without reliable f16/f128 math, where the
// gated test functions below are compiled out.
#![allow(unused_features)]
// `target_has_reliable_*` are not "known" configs since they are unstable.
#![expect(unexpected_cfgs)]

use std::hint::black_box;

fn main() {
    test_f32();
    test_f64();
    #[cfg(target_has_reliable_f16_math)]
    test_f16();
    #[cfg(target_has_reliable_f128_math)]
    test_f128();
}

fn test_f32() {
    // Exactly representable results are the same whether or not the
    // operation is fused.
    assert_eq!(black_box(2.0_f32).mul_add_relaxed(3.0, 4.0), 10.0);
    assert_eq!(black_box(1.0_f32).mul_add_relaxed(1.0, 1.0), 2.0);

    // `0.1 * 0.1` is inexact, so the fused (one rounding) and unfused (two
    // roundings) results differ; either is allowed.
    let r = black_box(0.1_f32).mul_add_relaxed(0.1, -0.01);
    assert!(r == 5.2154064e-10 || r == 9.313226e-10);

    // Edge cases behave like `a * b + c` regardless of fusion.
    assert!(black_box(f32::NAN).mul_add_relaxed(1.0, 1.0).is_nan());
    assert_eq!(black_box(f32::INFINITY).mul_add_relaxed(2.0, 1.0), f32::INFINITY);
    assert!(black_box(0.0_f32).mul_add_relaxed(f32::INFINITY, 1.0).is_nan());
}

fn test_f64() {
    assert_eq!(black_box(2.0_f64).mul_add_relaxed(3.0, 4.0), 10.0);
    assert_eq!(black_box(1.0_f64).mul_add_relaxed(1.0, 1.0), 2.0);

    let r = black_box(0.1_f64).mul_add_relaxed(0.1, -0.01);
    assert!(r == 9.020562075079397e-19 || r == 1.734723475976807e-18);

    assert!(black_box(f64::NAN).mul_add_relaxed(1.0, 1.0).is_nan());
    assert_eq!(black_box(f64::INFINITY).mul_add_relaxed(2.0, 1.0), f64::INFINITY);
    assert!(black_box(0.0_f64).mul_add_relaxed(f64::INFINITY, 1.0).is_nan());
}

#[cfg(target_has_reliable_f16_math)]
fn test_f16() {
    assert_eq!(black_box(2.0_f16).mul_add_relaxed(3.0, 4.0), 10.0);
    assert_eq!(black_box(1.0_f16).mul_add_relaxed(1.0, 1.0), 2.0);

    assert!(black_box(f16::NAN).mul_add_relaxed(1.0, 1.0).is_nan());
    assert_eq!(black_box(f16::INFINITY).mul_add_relaxed(2.0, 1.0), f16::INFINITY);
    assert!(black_box(0.0_f16).mul_add_relaxed(f16::INFINITY, 1.0).is_nan());
}

#[cfg(target_has_reliable_f128_math)]
fn test_f128() {
    assert_eq!(black_box(2.0_f128).mul_add_relaxed(3.0, 4.0), 10.0);
    assert_eq!(black_box(1.0_f128).mul_add_relaxed(1.0, 1.0), 2.0);

    assert!(black_box(f128::NAN).mul_add_relaxed(1.0, 1.0).is_nan());
    assert_eq!(black_box(f128::INFINITY).mul_add_relaxed(2.0, 1.0), f128::INFINITY);
    assert!(black_box(0.0_f128).mul_add_relaxed(f128::INFINITY, 1.0).is_nan());
}
