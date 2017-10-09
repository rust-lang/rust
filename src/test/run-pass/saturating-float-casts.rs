// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z saturating-float-casts

#![feature(test, i128, i128_type, stmt_expr_attributes)]
#![deny(overflowing_literals)]
extern crate test;

use std::{f32, f64};
use std::{u8, i8, u16, i16, u32, i32, u64, i64, u128, i128};
use test::black_box;

macro_rules! test {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => (
        // black_box disables constant evaluation to test run-time conversions:
        assert_eq!(black_box::<$src_ty>($val) as $dest_ty, $expected,
                    "run time {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        // ... whereas this variant triggers constant evaluation:
        {
            const X: $src_ty = $val;
            const Y: $dest_ty = X as $dest_ty;
            assert_eq!(Y, $expected,
                        "const eval {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        }
    );

    ($fval:expr, f* -> $ity:ident, $ival:expr) => (
        test!($fval, f32 -> $ity, $ival);
        test!($fval, f64 -> $ity, $ival);
    )
}

macro_rules! common_fptoi_tests {
    ($fty:ident -> $($ity:ident)+) => ({ $(
        test!($fty::NAN, $fty -> $ity, 0);
        test!($fty::INFINITY, $fty -> $ity, $ity::MAX);
        test!($fty::NEG_INFINITY, $fty -> $ity, $ity::MIN);
        // These two tests are not solely float->int tests, in particular the latter relies on
        // `u128::MAX as f32` not being UB. But that's okay, since this file tests int->float
        // as well, the test is just slightly misplaced.
        test!($ity::MIN as $fty, $fty -> $ity, $ity::MIN);
        test!($ity::MAX as $fty, $fty -> $ity, $ity::MAX);
        test!(0., $fty -> $ity, 0);
        test!($fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.9, $fty -> $ity, 0);
        test!(1., $fty -> $ity, 1);
        test!(42., $fty -> $ity, 42);
    )+ });

    (f* -> $($ity:ident)+) => ({
        common_fptoi_tests!(f32 -> $($ity)+);
        common_fptoi_tests!(f64 -> $($ity)+);
    })
}

macro_rules! fptoui_tests {
    ($fty: ident -> $($ity: ident)+) => ({ $(
        test!(-0., $fty -> $ity, 0);
        test!(-$fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.99999994, $fty -> $ity, 0);
        test!(-1., $fty -> $ity, 0);
        test!(-100., $fty -> $ity, 0);
        test!(#[allow(overflowing_literals)] -1e50, $fty -> $ity, 0);
        test!(#[allow(overflowing_literals)] -1e130, $fty -> $ity, 0);
    )+ });

    (f* -> $($ity:ident)+) => ({
        fptoui_tests!(f32 -> $($ity)+);
        fptoui_tests!(f64 -> $($ity)+);
    })
}

pub fn main() {
    common_fptoi_tests!(f* -> i8 i16 i32 i64 i128 u8 u16 u32 u64 u128);
    fptoui_tests!(f* -> u8 u16 u32 u64 u128);

    // The following tests cover edge cases for some integer types.

    // u8
    test!(254., f* -> u8, 254);
    test!(256., f* -> u8, 255);

    // i8
    test!(-127., f* -> i8, -127);
    test!(-129., f* -> i8, -128);
    test!(126., f* -> i8, 126);
    test!(128., f* -> i8, 127);

    // i32
    // -2147483648. is i32::MIN (exactly)
    test!(-2147483648., f* -> i32, i32::MIN);
    // 2147483648. is i32::MAX rounded up
    test!(2147483648., f32 -> i32, 2147483647);
    // With 24 significand bits, floats with magnitude in [2^30 + 1, 2^31] are rounded to
    // multiples of 2^7. Therefore, nextDown(round(i32::MAX)) is 2^31 - 128:
    test!(2147483520., f32 -> i32, 2147483520);
    // Similarly, nextUp(i32::MIN) is i32::MIN + 2^8 and nextDown(i32::MIN) is i32::MIN - 2^7
    test!(-2147483904., f* -> i32, i32::MIN);
    test!(-2147483520., f* -> i32, -2147483520);

    // u32 -- round(MAX) and nextUp(round(MAX))
    test!(4294967040., f* -> u32, 4294967040);
    test!(4294967296., f* -> u32, 4294967295);

    // u128
    // # float->int
    test!(f32::MAX, f32 -> u128, 0xffffff00000000000000000000000000);
    // nextDown(f32::MAX) = 2^128 - 2 * 2^104
    const SECOND_LARGEST_F32: f32 = 340282326356119256160033759537265639424.;
    test!(SECOND_LARGEST_F32, f32 -> u128, 0xfffffe00000000000000000000000000);
    // # int->float
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

    test!(!0, u128 -> f32, f32::INFINITY);

    // u128->f64 should not be affected by the u128->f32 checks
    test!(0xffffff80000000000000000000000000, u128 -> f64,
          340282356779733661637539395458142568448.0);
    test!(u128::MAX, u128 -> f64, 340282366920938463463374607431768211455.0);
}
