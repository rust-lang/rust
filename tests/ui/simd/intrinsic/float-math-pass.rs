//@ run-pass
//@ ignore-emscripten
//@ ignore-android

// FIXME: this test fails on arm-android because the NDK version 14 is too old.
// It needs at least version 18. We disable it on all android build bots because
// there is no way in compile-test to disable it for an (arch,os) pair.

// Test that the simd floating-point math intrinsics produce correct results.

#![feature(repr_simd, intrinsics)]
#![allow(non_camel_case_types)]

#[repr(simd)]
#[derive(Copy, Clone, PartialEq, Debug)]
struct f32x4(pub [f32; 4]);

#[rustc_intrinsic]
unsafe fn simd_fsqrt<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_fabs<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_fsin<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_fcos<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_fexp<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_fexp2<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_fma<T>(x: T, y: T, z: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_relaxed_fma<T>(x: T, y: T, z: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_flog<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_flog10<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_flog2<T>(x: T) -> T;

// rounding functions
#[rustc_intrinsic]
unsafe fn simd_ceil<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_floor<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_round<T>(x: T) -> T;

#[rustc_intrinsic]
unsafe fn simd_trunc<T>(x: T) -> T;

macro_rules! assert_approx_eq_f32 {
    ($a:expr, $b:expr) => {{
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6, "{} is not approximately equal to {}", *a, *b);
    }};
}
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {{
        let a = $a;
        let b = $b;
        assert_approx_eq_f32!(a.0[0], b.0[0]);
        assert_approx_eq_f32!(a.0[1], b.0[1]);
        assert_approx_eq_f32!(a.0[2], b.0[2]);
        assert_approx_eq_f32!(a.0[3], b.0[3]);
    }};
}

fn main() {
    let x = f32x4([1.0, 1.0, 1.0, 1.0]);
    let y = f32x4([-1.0, -1.0, -1.0, -1.0]);
    let z = f32x4([0.0, 0.0, 0.0, 0.0]);

    let h = f32x4([0.5, 0.5, 0.5, 0.5]);

    unsafe {
        let r = simd_fabs(y);
        assert_approx_eq!(x, r);

        let r = simd_fcos(z);
        assert_approx_eq!(x, r);

        let r = simd_fexp(z);
        assert_approx_eq!(x, r);

        let r = simd_fexp2(z);
        assert_approx_eq!(x, r);

        let r = simd_fma(x, h, h);
        assert_approx_eq!(x, r);

        let r = simd_relaxed_fma(x, h, h);
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

        // rounding functions
        let r = simd_floor(h);
        assert_eq!(z, r);

        let r = simd_ceil(h);
        assert_eq!(x, r);

        let r = simd_round(h);
        assert_eq!(x, r);

        let r = simd_trunc(h);
        assert_eq!(z, r);
    }
}
