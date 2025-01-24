#![feature(core_intrinsics, portable_simd)]
use std::intrinsics::simd::simd_relaxed_fma;
use std::intrinsics::{fmuladdf32, fmuladdf64};
use std::simd::prelude::*;

fn ensure_both_happen(f: impl Fn() -> bool) -> bool {
    let mut saw_true = false;
    let mut saw_false = false;
    for _ in 0..50 {
        let b = f();
        if b {
            saw_true = true;
        } else {
            saw_false = true;
        }
        if saw_true && saw_false {
            return true;
        }
    }
    false
}

fn main() {
    assert!(
        ensure_both_happen(|| {
            let a = std::hint::black_box(0.1_f64);
            let b = std::hint::black_box(0.2);
            let c = std::hint::black_box(-a * b);
            // It is unspecified whether the following operation is fused or not. The
            // following evaluates to 0.0 if unfused, and nonzero (-1.66e-18) if fused.
            let x = unsafe { fmuladdf64(a, b, c) };
            x == 0.0
        }),
        "`fmuladdf64` failed to be evaluated as both fused and unfused"
    );

    assert!(
        ensure_both_happen(|| {
            let a = std::hint::black_box(0.1_f32);
            let b = std::hint::black_box(0.2);
            let c = std::hint::black_box(-a * b);
            // It is unspecified whether the following operation is fused or not. The
            // following evaluates to 0.0 if unfused, and nonzero (-8.1956386e-10) if fused.
            let x = unsafe { fmuladdf32(a, b, c) };
            x == 0.0
        }),
        "`fmuladdf32` failed to be evaluated as both fused and unfused"
    );

    assert!(
        ensure_both_happen(|| {
            let a = f32x4::splat(std::hint::black_box(0.1));
            let b = f32x4::splat(std::hint::black_box(0.2));
            let c = std::hint::black_box(-a * b);
            let x = unsafe { simd_relaxed_fma(a, b, c) };
            // Whether we fuse or not is a per-element decision, so sometimes these should be
            // the same and sometimes not.
            x[0] == x[1]
        }),
        "`simd_relaxed_fma` failed to be evaluated as both fused and unfused"
    );

    assert!(
        ensure_both_happen(|| {
            let a = f64x4::splat(std::hint::black_box(0.1));
            let b = f64x4::splat(std::hint::black_box(0.2));
            let c = std::hint::black_box(-a * b);
            let x = unsafe { simd_relaxed_fma(a, b, c) };
            // Whether we fuse or not is a per-element decision, so sometimes these should be
            // the same and sometimes not.
            x[0] == x[1]
        }),
        "`simd_relaxed_fma` failed to be evaluated as both fused and unfused"
    );
}
