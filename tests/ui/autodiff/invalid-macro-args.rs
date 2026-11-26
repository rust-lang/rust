//! Regression test for <https://github.com/rust-lang/rust/issues/150326>,
//! <https://github.com/rust-lang/rust/issues/150327> and
//! <https://github.com/rust-lang/rust/issues/150329>.
//! Invalid arguments to the autodiff macros used to ICE during expansion.

#![feature(autodiff)]

use std::autodiff::{autodiff_forward, autodiff_reverse};

// A return type, but no activities at all.
#[autodiff_reverse(df1)]
//~^ ERROR expected 1 activities, but found 0
fn f1() -> f64 {
    0.0
}

// The name has a leading `::`.
#[autodiff_forward(::df2)]
//~^ ERROR expected an identifier for the generated function name
fn f2() {}

// The name is a float literal.
#[autodiff_reverse(2.0f)]
//~^ ERROR invalid suffix `f` for float literal
//~| ERROR expected an identifier for the generated function name
fn f3() {}

// An activity that is a float literal.
#[autodiff_reverse(df4, 2.0)]
//~^ ERROR did not recognize Activity: `2.0`
fn f4(x: f64) {}

// An activity with a path.
#[autodiff_reverse(df5, ::Active)]
//~^ ERROR did not recognize Activity: `::Active`
fn f5(x: f64) {}

fn main() {}
