// Compiler:
//
// Run-time:
//   status: 0

#![feature(portable_simd)]

use std::hint::black_box;
use std::simd::prelude::*;

fn test_saturating_add() {
    let values = i32x4::from_array([i32::MIN, -2, 3, i32::MAX]);
    let ones = i32x4::splat(1);
    assert_eq!(
        black_box(values).saturating_add(black_box(ones)).to_array(),
        [i32::MIN + 1, -1, 4, i32::MAX]
    );

    let values = u32x4::from_array([0, 2, 3, u32::MAX]);
    let ones = u32x4::splat(1);
    assert_eq!(black_box(values).saturating_add(black_box(ones)).to_array(), [1, 3, 4, u32::MAX]);
}

fn test_saturating_sub() {
    let values = i32x4::from_array([i32::MIN, -2, 3, i32::MAX]);
    let zero = i32x4::splat(0);
    assert_eq!(
        black_box(zero).saturating_sub(black_box(values)).to_array(),
        [i32::MAX, 2, -3, i32::MIN + 1]
    );
    assert_eq!(black_box(values).saturating_neg().to_array(), [i32::MAX, 2, -3, i32::MIN + 1]);
    assert_eq!(black_box(values).saturating_abs().to_array(), [i32::MAX, 2, 3, i32::MAX]);

    let values = i32x4::from_array([i32::MIN, -2, 3, i32::MAX]);
    let ones = i32x4::splat(1);
    assert_eq!(
        black_box(values).saturating_sub(black_box(ones)).to_array(),
        [i32::MIN, -3, 2, i32::MAX - 1]
    );

    let values = u32x4::from_array([0, 2, 3, u32::MAX]);
    let ones = u32x4::splat(1);
    assert_eq!(
        black_box(values).saturating_sub(black_box(ones)).to_array(),
        [0, 1, 2, u32::MAX - 1]
    );
}

fn test_float_cast() {
    let floats = f32x4::from_array([1.9, -4.5, f32::INFINITY, f32::NAN]);
    assert_eq!(black_box(floats).cast::<i32>().to_array(), [1, -4, i32::MAX, 0]);

    let floats = f32x4::from_array([f32::NEG_INFINITY, 1e20, -1e20, -0.0]);
    assert_eq!(black_box(floats).cast::<i32>().to_array(), [i32::MIN, i32::MAX, i32::MIN, 0]);

    let floats = f32x4::from_array([-1.0, 3.7, f32::NAN, 1e20]);
    assert_eq!(black_box(floats).cast::<u32>().to_array(), [0, 3, 0, u32::MAX]);

    let floats = f64x4::from_array([-1.5, 2.5, f64::NAN, f64::INFINITY]);
    assert_eq!(black_box(floats).cast::<i64>().to_array(), [-1, 2, 0, i64::MAX]);
}

fn test_arith_offset() {
    let values = [10i32, 11, 12, 13, 14, 15, 16, 17];
    let indices = usizex4::from_array([7, 5, 3, 1]);
    assert_eq!(
        i32x4::gather_or_default(black_box(&values), black_box(indices)).to_array(),
        [17, 15, 13, 11]
    );

    let mut destination = [0i32; 8];
    i32x4::from_array([1, 2, 3, 4]).scatter(black_box(&mut destination), black_box(indices));
    assert_eq!(destination, [0, 4, 0, 3, 0, 2, 0, 1]);
}

fn main() {
    test_saturating_add();
    test_saturating_sub();
    test_float_cast();
    test_arith_offset();
}
