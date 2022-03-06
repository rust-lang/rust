#![feature(portable_simd, platform_intrinsics)]
use std::simd::*;

fn simd_ops_f32() {
    let a = f32x4::splat(10.0);
    let b = f32x4::from_array([1.0, 2.0, 3.0, -4.0]);
    assert_eq!(-b, f32x4::from_array([-1.0, -2.0, -3.0, 4.0]));
    assert_eq!(a + b, f32x4::from_array([11.0, 12.0, 13.0, 6.0]));
    assert_eq!(a - b, f32x4::from_array([9.0, 8.0, 7.0, 14.0]));
    assert_eq!(a * b, f32x4::from_array([10.0, 20.0, 30.0, -40.0]));
    assert_eq!(b / a, f32x4::from_array([0.1, 0.2, 0.3, -0.4]));
    assert_eq!(a / f32x4::splat(2.0), f32x4::splat(5.0));
    assert_eq!(a % b, f32x4::from_array([0.0, 0.0, 1.0, 2.0]));
    assert_eq!(b.abs(), f32x4::from_array([1.0, 2.0, 3.0, 4.0]));

    // FIXME use Mask::from_array once simd_cast is implemented.
    assert_eq!(a.lanes_eq(f32x4::splat(5.0)*b), Mask::from_int(i32x4::from_array([0, -1, 0, 0])));
    assert_eq!(a.lanes_ne(f32x4::splat(5.0)*b), Mask::from_int(i32x4::from_array([-1, 0, -1, -1])));
    assert_eq!(a.lanes_le(f32x4::splat(5.0)*b), Mask::from_int(i32x4::from_array([0, -1, -1, 0])));
    assert_eq!(a.lanes_lt(f32x4::splat(5.0)*b), Mask::from_int(i32x4::from_array([0, 0, -1, 0])));
    assert_eq!(a.lanes_ge(f32x4::splat(5.0)*b), Mask::from_int(i32x4::from_array([-1, -1, 0, -1])));
    assert_eq!(a.lanes_gt(f32x4::splat(5.0)*b), Mask::from_int(i32x4::from_array([-1, 0, 0, -1])));
}

fn simd_ops_f64() {
    let a = f64x4::splat(10.0);
    let b = f64x4::from_array([1.0, 2.0, 3.0, -4.0]);
    assert_eq!(-b, f64x4::from_array([-1.0, -2.0, -3.0, 4.0]));
    assert_eq!(a + b, f64x4::from_array([11.0, 12.0, 13.0, 6.0]));
    assert_eq!(a - b, f64x4::from_array([9.0, 8.0, 7.0, 14.0]));
    assert_eq!(a * b, f64x4::from_array([10.0, 20.0, 30.0, -40.0]));
    assert_eq!(b / a, f64x4::from_array([0.1, 0.2, 0.3, -0.4]));
    assert_eq!(a / f64x4::splat(2.0), f64x4::splat(5.0));
    assert_eq!(a % b, f64x4::from_array([0.0, 0.0, 1.0, 2.0]));
    assert_eq!(b.abs(), f64x4::from_array([1.0, 2.0, 3.0, 4.0]));

    // FIXME use Mask::from_array once simd_cast is implemented.
    assert_eq!(a.lanes_eq(f64x4::splat(5.0)*b), Mask::from_int(i64x4::from_array([0, -1, 0, 0])));
    assert_eq!(a.lanes_ne(f64x4::splat(5.0)*b), Mask::from_int(i64x4::from_array([-1, 0, -1, -1])));
    assert_eq!(a.lanes_le(f64x4::splat(5.0)*b), Mask::from_int(i64x4::from_array([0, -1, -1, 0])));
    assert_eq!(a.lanes_lt(f64x4::splat(5.0)*b), Mask::from_int(i64x4::from_array([0, 0, -1, 0])));
    assert_eq!(a.lanes_ge(f64x4::splat(5.0)*b), Mask::from_int(i64x4::from_array([-1, -1, 0, -1])));
    assert_eq!(a.lanes_gt(f64x4::splat(5.0)*b), Mask::from_int(i64x4::from_array([-1, 0, 0, -1])));
}

fn simd_ops_i32() {
    let a = i32x4::splat(10);
    let b = i32x4::from_array([1, 2, 3, -4]);
    assert_eq!(-b, i32x4::from_array([-1, -2, -3, 4]));
    assert_eq!(a + b, i32x4::from_array([11, 12, 13, 6]));
    assert_eq!(a - b, i32x4::from_array([9, 8, 7, 14]));
    assert_eq!(a * b, i32x4::from_array([10, 20, 30, -40]));
    assert_eq!(a / b, i32x4::from_array([10, 5, 3, -2]));
    assert_eq!(a / i32x4::splat(2), i32x4::splat(5));
    assert_eq!(i32x2::splat(i32::MIN) / i32x2::splat(-1), i32x2::splat(i32::MIN));
    assert_eq!(a % b, i32x4::from_array([0, 0, 1, 2]));
    assert_eq!(i32x2::splat(i32::MIN) % i32x2::splat(-1), i32x2::splat(0));
    assert_eq!(b << i32x4::splat(2), i32x4::from_array([4, 8, 12, -16]));
    assert_eq!(b >> i32x4::splat(1), i32x4::from_array([0, 1, 1, -2]));
    assert_eq!(b & i32x4::splat(2), i32x4::from_array([0, 2, 2, 0]));
    assert_eq!(b | i32x4::splat(2), i32x4::from_array([3, 2, 3, -2]));
    assert_eq!(b ^ i32x4::splat(2), i32x4::from_array([3, 0, 1, -2]));

    // FIXME use Mask::from_array once simd_cast is implemented.
    assert_eq!(a.lanes_eq(i32x4::splat(5)*b), Mask::from_int(i32x4::from_array([0, -1, 0, 0])));
    assert_eq!(a.lanes_ne(i32x4::splat(5)*b), Mask::from_int(i32x4::from_array([-1, 0, -1, -1])));
    assert_eq!(a.lanes_le(i32x4::splat(5)*b), Mask::from_int(i32x4::from_array([0, -1, -1, 0])));
    assert_eq!(a.lanes_lt(i32x4::splat(5)*b), Mask::from_int(i32x4::from_array([0, 0, -1, 0])));
    assert_eq!(a.lanes_ge(i32x4::splat(5)*b), Mask::from_int(i32x4::from_array([-1, -1, 0, -1])));
    assert_eq!(a.lanes_gt(i32x4::splat(5)*b), Mask::from_int(i32x4::from_array([-1, 0, 0, -1])));

    assert_eq!(a.horizontal_and(), 10);
    assert_eq!(b.horizontal_and(), 0);
    assert_eq!(a.horizontal_or(), 10);
    assert_eq!(b.horizontal_or(), -1);
    assert_eq!(a.horizontal_xor(), 0);
    assert_eq!(b.horizontal_xor(), -4);
    assert_eq!(a.horizontal_sum(), 40);
    assert_eq!(b.horizontal_sum(), 2);
    assert_eq!(a.horizontal_product(), 100*100);
    assert_eq!(b.horizontal_product(), 6*-4);
}

fn simd_intrinsics() {
    extern "platform-intrinsic" {
        fn simd_eq<T, U>(x: T, y: T) -> U;
        fn simd_reduce_any<T>(x: T) -> bool;
        fn simd_reduce_all<T>(x: T) -> bool;
        fn simd_select<M, T>(m: M, yes: T, no: T) -> T;
    }
    unsafe {
        // Make sure simd_eq returns all-1 for `true`
        let a = i32x4::splat(10);
        let b = i32x4::from_array([1, 2, 10, 4]);
        let c: i32x4 = simd_eq(a, b);
        assert_eq!(c, i32x4::from_array([0, 0, -1, 0]));

        assert!(!simd_reduce_any(i32x4::splat(0)));
        assert!(simd_reduce_any(i32x4::splat(-1)));
        assert!(simd_reduce_any(i32x2::from_array([0, -1])));
        assert!(!simd_reduce_all(i32x4::splat(0)));
        assert!(simd_reduce_all(i32x4::splat(-1)));
        assert!(!simd_reduce_all(i32x2::from_array([0, -1])));

        assert_eq!(simd_select(i8x4::from_array([0, -1, -1, 0]), a, b), i32x4::from_array([1, 10, 10, 4]));
        assert_eq!(simd_select(i8x4::from_array([0, -1, -1, 0]), b, a), i32x4::from_array([10, 2, 10, 10]));
    }
}

fn main() {
    simd_ops_f32();
    simd_ops_f64();
    simd_ops_i32();
    simd_intrinsics();
}
