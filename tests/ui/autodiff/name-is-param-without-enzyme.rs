//! Regression test for <https://github.com/rust-lang/rust/issues/150330> without Enzyme.
//! Codegen reports the missing flag and goes on to read the generated function from the call.
//! That call used to name the parameter `x` instead of the function, which ICEd.

//@ build-fail
//@ compile-flags: -Clto=fat
//@ no-prefer-dynamic
//@ ignore-backends: gcc

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(x, Active, Active)]
fn f(x: f64) -> f64 {
    x * x
}

fn main() {
    let _ = x(1.0, 1.0);
}

//~? ERROR using the autodiff feature requires -Z autodiff=Enable
