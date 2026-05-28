#![feature(core_intrinsics, portable_simd)]
use std::intrinsics::simd::simd_relaxed_fma;
use std::intrinsics::{fmuladdf32, fmuladdf64};
use std::simd::prelude::*;

#[path = "../../utils/mod.rs"]
mod utils;
use utils::check_nondet;

fn main() {
    check_nondet(|| {
        let a = std::hint::black_box(0.1_f64);
        let b = std::hint::black_box(0.2);
        let c = std::hint::black_box(-a * b);
        // It is unspecified whether the following operation is fused or not. The
        // following evaluates to 0.0 if unfused, and nonzero (-1.66e-18) if fused.
        let x = fmuladdf64(a, b, c);
        x == 0.0
    });

    check_nondet(|| {
        let a = std::hint::black_box(0.1_f32);
        let b = std::hint::black_box(0.2);
        let c = std::hint::black_box(-a * b);
        // It is unspecified whether the following operation is fused or not. The
        // following evaluates to 0.0 if unfused, and nonzero (-8.1956386e-10) if fused.
        let x = fmuladdf32(a, b, c);
        x == 0.0
    });

    check_nondet(|| {
        let a = f32x4::splat(std::hint::black_box(0.1));
        let b = f32x4::splat(std::hint::black_box(0.2));
        let c = std::hint::black_box(-a * b);
        let x = unsafe { simd_relaxed_fma(a, b, c) };
        // Whether we fuse or not is a per-element decision, so sometimes these should be
        // the same and sometimes not.
        x[0] == x[1]
    });

    check_nondet(|| {
        let a = f64x4::splat(std::hint::black_box(0.1));
        let b = f64x4::splat(std::hint::black_box(0.2));
        let c = std::hint::black_box(-a * b);
        let x = unsafe { simd_relaxed_fma(a, b, c) };
        // Whether we fuse or not is a per-element decision, so sometimes these should be
        // the same and sometimes not.
        x[0] == x[1]
    });
}
