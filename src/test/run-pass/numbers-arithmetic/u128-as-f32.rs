// run-pass
// ignore-emscripten u128 not supported

#![feature(test)]
#![deny(overflowing_literals)]
extern crate test;

use std::f32;
use std::u128;
use test::black_box;

macro_rules! test {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => ({
        {
            const X: $src_ty = $val;
            const Y: $dest_ty = X as $dest_ty;
            assert_eq!(Y, $expected,
                        "const eval {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        }
        // black_box disables constant evaluation to test run-time conversions:
        assert_eq!(black_box::<$src_ty>($val) as $dest_ty, $expected,
                    "run-time {} -> {}", stringify!($src_ty), stringify!($dest_ty));
    });
}

pub fn main() {
    // nextDown(f32::MAX) = 2^128 - 2 * 2^104
    const SECOND_LARGEST_F32: f32 = 340282326356119256160033759537265639424.;

    // f32::MAX - 0.5 ULP and smaller should be rounded down
    test!(0xfffffe00000000000000000000000000, u128 -> f32, SECOND_LARGEST_F32);
    test!(0xfffffe7fffffffffffffffffffffffff, u128 -> f32, SECOND_LARGEST_F32);
    test!(0xfffffe80000000000000000000000000, u128 -> f32, SECOND_LARGEST_F32);
    // numbers within < 0.5 ULP of f32::MAX it should be rounded to f32::MAX
    test!(0xfffffe80000000000000000000000001, u128 -> f32, f32::MAX);
    test!(0xfffffeffffffffffffffffffffffffff, u128 -> f32, f32::MAX);
    test!(0xffffff00000000000000000000000000, u128 -> f32, f32::MAX);
    test!(0xffffff00000000000000000000000001, u128 -> f32, f32::MAX);
    test!(0xffffff7fffffffffffffffffffffffff, u128 -> f32, f32::MAX);
    // f32::MAX + 0.5 ULP and greater should be rounded to infinity
    test!(0xffffff80000000000000000000000000, u128 -> f32, f32::INFINITY);
    test!(0xffffff80000000f00000000000000000, u128 -> f32, f32::INFINITY);
    test!(0xffffff87ffffffffffffffff00000001, u128 -> f32, f32::INFINITY);

    // u128->f64 should not be affected by the u128->f32 checks
    test!(0xffffff80000000000000000000000000, u128 -> f64,
          340282356779733661637539395458142568448.0);
    test!(u128::MAX, u128 -> f64, 340282366920938463463374607431768211455.0);
}
