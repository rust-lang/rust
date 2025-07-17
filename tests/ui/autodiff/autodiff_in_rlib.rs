#![feature(autodiff)]
#![crate_type = "rlib"]
//@ needs-enzyme
//@ compile-flags: -Zautodiff=Enable -C opt-level=3  -Clto=fat
//@ build-fail

// We test that we fail to compile if a user applied an autodiff_ macro in src/lib.rs,
// since autodiff doesn't work in libraries yet. In the past we used to just return zeros in the
// autodiffed functions, which is obviously confusing and wrong, so erroring is an improvement.

use std::autodiff::autodiff_reverse;
//~? ERROR: using the autodiff feature with library builds is not yet supported

#[autodiff_reverse(d_square, Duplicated, Active)]
pub fn square(x: &f64) -> f64 {
    *x * *x
}
