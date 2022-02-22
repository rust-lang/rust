#![feature(portable_simd, platform_intrinsics)]
use std::simd::*;

fn simd_ops_f32() {
    let a = f32x4::splat(10.0);
    let b = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(a + b, f32x4::from_array([11.0, 12.0, 13.0, 14.0]));
    assert_eq!(a - b, f32x4::from_array([9.0, 8.0, 7.0, 6.0]));
    assert_eq!(a * b, f32x4::from_array([10.0, 20.0, 30.0, 40.0]));
    assert_eq!(b / a, f32x4::from_array([0.1, 0.2, 0.3, 0.4]));
    assert_eq!(a / f32x4::splat(2.0), f32x4::splat(5.0));
    assert_eq!(a % b, f32x4::from_array([0.0, 0.0, 1.0, 2.0]));
}

fn simd_ops_i32() {
    let a = i32x4::splat(10);
    let b = i32x4::from_array([1, 2, 3, 4]);
    assert_eq!(a + b, i32x4::from_array([11, 12, 13, 14]));
    assert_eq!(a - b, i32x4::from_array([9, 8, 7, 6]));
    assert_eq!(a * b, i32x4::from_array([10, 20, 30, 40]));
    assert_eq!(a / b, i32x4::from_array([10, 5, 3, 2]));
    assert_eq!(a / i32x4::splat(2), i32x4::splat(5));
    assert_eq!(a % b, i32x4::from_array([0, 0, 1, 2]));
    assert_eq!(b << i32x4::splat(2), i32x4::from_array([4, 8, 12, 16]));
    assert_eq!(b >> i32x4::splat(1), i32x4::from_array([0, 1, 1, 2]));
    assert_eq!(b & i32x4::splat(2), i32x4::from_array([0, 2, 2, 0]));
    assert_eq!(b | i32x4::splat(2), i32x4::from_array([3, 2, 3, 6]));
}

fn simd_intrinsics() {
    extern "platform-intrinsic" {
        pub(crate) fn simd_eq<T, U>(x: T, y: T) -> U;
        pub(crate) fn simd_reduce_any<T>(x: T) -> bool;
    }

    // Make sure simd_eq returns all-1 for `true`
    let a = i32x4::splat(10);
    let b = i32x4::from_array([1, 2, 10, 4]);
    let c: i32x4 = unsafe { simd_eq(a, b) };
    assert_eq!(c, i32x4::from_array([0, 0, -1, 0]));

    unsafe {
        assert!(!simd_reduce_any(i32x4::splat(0)));
        assert!(simd_reduce_any(i32x4::splat(-1)));
    }
}

fn main() {
    simd_ops_f32();
    simd_ops_i32();
    simd_intrinsics();
}
