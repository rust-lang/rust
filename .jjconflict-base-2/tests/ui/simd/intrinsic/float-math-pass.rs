//@ run-pass
//@ ignore-emscripten
//@ ignore-android
//@ compile-flags: --cfg minisimd_const

// FIXME: this test fails on arm-android because the NDK version 14 is too old.
// It needs at least version 18. We disable it on all android build bots because
// there is no way in compile-test to disable it for an (arch,os) pair.

// Test that the simd floating-point math intrinsics produce correct results.

#![feature(repr_simd, core_intrinsics, const_trait_impl, const_cmp, const_index)]
#![allow(non_camel_case_types)]

#[path = "../../../auxiliary/minisimd.rs"]
mod minisimd;
use minisimd::*;

use std::intrinsics::simd::*;

macro_rules! assert_approx_eq_f32 {
    ($a:expr, $b:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!(
            (*a - *b).abs() < 1.0e-6,
            concat!(stringify!($a), " is not approximately equal to ", stringify!($b))
        );
    }};
}
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        assert_approx_eq_f32!(a[0], b[0]);
        assert_approx_eq_f32!(a[1], b[1]);
        assert_approx_eq_f32!(a[2], b[2]);
        assert_approx_eq_f32!(a[3], b[3]);
    }};
}

const fn simple_math() {
    let x = f32x4::from_array([1.0, 1.0, 1.0, 1.0]);
    let y = f32x4::from_array([-1.0, -1.0, -1.0, -1.0]);
    let z = f32x4::from_array([0.0, 0.0, 0.0, 0.0]);

    let h = f32x4::from_array([0.5, 0.5, 0.5, 0.5]);

    unsafe {
        let r = simd_fabs(y);
        assert_eq!(x, r);

        // rounding functions
        let r = simd_floor(h);
        assert_eq!(z, r);

        let r = simd_ceil(h);
        assert_eq!(x, r);

        let r = simd_round(h);
        assert_eq!(x, r);

        let r = simd_round_ties_even(h);
        assert_eq!(z, r);

        let r = simd_trunc(h);
        assert_eq!(z, r);

        let r = simd_fma(x, h, h);
        assert_approx_eq!(x, r);

        let r = simd_relaxed_fma(x, h, h);
        assert_approx_eq!(x, r);
    }
}

fn special_math() {
    let x = f32x4::from_array([1.0, 1.0, 1.0, 1.0]);
    let z = f32x4::from_array([0.0, 0.0, 0.0, 0.0]);

    unsafe {
        let r = simd_fcos(z);
        assert_approx_eq!(x, r);

        let r = simd_fexp(z);
        assert_approx_eq!(x, r);

        let r = simd_fexp2(z);
        assert_approx_eq!(x, r);

        let r = simd_fsqrt(x);
        assert_approx_eq!(x, r);

        let r = simd_flog(x);
        assert_approx_eq!(z, r);

        let r = simd_flog2(x);
        assert_approx_eq!(z, r);

        let r = simd_flog10(x);
        assert_approx_eq!(z, r);

        let r = simd_fsin(z);
        assert_approx_eq!(z, r);
    }
}

fn main() {
    const { simple_math() };
    simple_math();
    special_math();
}
