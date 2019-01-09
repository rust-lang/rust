// run-pass
// Tests saturating float->int casts. See u128-as-f32.rs for the opposite direction.
// compile-flags: -Z saturating-float-casts

#![feature(test, stmt_expr_attributes)]
#![deny(overflowing_literals)]
extern crate test;

use std::{f32, f64};
use std::{u8, i8, u16, i16, u32, i32, u64, i64};
#[cfg(not(target_os="emscripten"))]
use std::{u128, i128};
use test::black_box;

macro_rules! test {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => (
        // black_box disables constant evaluation to test run-time conversions:
        assert_eq!(black_box::<$src_ty>($val) as $dest_ty, $expected,
                    "run-time {} -> {}", stringify!($src_ty), stringify!($dest_ty));
    );

    ($fval:expr, f* -> $ity:ident, $ival:expr) => (
        test!($fval, f32 -> $ity, $ival);
        test!($fval, f64 -> $ity, $ival);
    )
}

// This macro tests const eval in addition to run-time evaluation.
// If and when saturating casts are adopted, this macro should be merged with test!() to ensure
// that run-time and const eval agree on inputs that currently trigger a const eval error.
macro_rules! test_c {
    ($val:expr, $src_ty:ident -> $dest_ty:ident, $expected:expr) => ({
        test!($val, $src_ty -> $dest_ty, $expected);
        {
            const X: $src_ty = $val;
            const Y: $dest_ty = X as $dest_ty;
            assert_eq!(Y, $expected,
                        "const eval {} -> {}", stringify!($src_ty), stringify!($dest_ty));
        }
    });

    ($fval:expr, f* -> $ity:ident, $ival:expr) => (
        test_c!($fval, f32 -> $ity, $ival);
        test_c!($fval, f64 -> $ity, $ival);
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
        test_c!(0., $fty -> $ity, 0);
        test_c!($fty::MIN_POSITIVE, $fty -> $ity, 0);
        test!(-0.9, $fty -> $ity, 0);
        test_c!(1., $fty -> $ity, 1);
        test_c!(42., $fty -> $ity, 42);
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
    common_fptoi_tests!(f* -> i8 i16 i32 i64 u8 u16 u32 u64);
    fptoui_tests!(f* -> u8 u16 u32 u64);
    // FIXME emscripten does not support i128
    #[cfg(not(target_os="emscripten"))] {
        common_fptoi_tests!(f* -> i128 u128);
        fptoui_tests!(f* -> u128);
    }

    // The following tests cover edge cases for some integer types.

    // # u8
    test_c!(254., f* -> u8, 254);
    test!(256., f* -> u8, 255);

    // # i8
    test_c!(-127., f* -> i8, -127);
    test!(-129., f* -> i8, -128);
    test_c!(126., f* -> i8, 126);
    test!(128., f* -> i8, 127);

    // # i32
    // -2147483648. is i32::MIN (exactly)
    test_c!(-2147483648., f* -> i32, i32::MIN);
    // 2147483648. is i32::MAX rounded up
    test!(2147483648., f32 -> i32, 2147483647);
    // With 24 significand bits, floats with magnitude in [2^30 + 1, 2^31] are rounded to
    // multiples of 2^7. Therefore, nextDown(round(i32::MAX)) is 2^31 - 128:
    test_c!(2147483520., f32 -> i32, 2147483520);
    // Similarly, nextUp(i32::MIN) is i32::MIN + 2^8 and nextDown(i32::MIN) is i32::MIN - 2^7
    test!(-2147483904., f* -> i32, i32::MIN);
    test_c!(-2147483520., f* -> i32, -2147483520);

    // # u32
    // round(MAX) and nextUp(round(MAX))
    test_c!(4294967040., f* -> u32, 4294967040);
    test!(4294967296., f* -> u32, 4294967295);

    // # u128
    #[cfg(not(target_os="emscripten"))]
    {
        // float->int:
        test_c!(f32::MAX, f32 -> u128, 0xffffff00000000000000000000000000);
        // nextDown(f32::MAX) = 2^128 - 2 * 2^104
        const SECOND_LARGEST_F32: f32 = 340282326356119256160033759537265639424.;
        test_c!(SECOND_LARGEST_F32, f32 -> u128, 0xfffffe00000000000000000000000000);
    }
}
