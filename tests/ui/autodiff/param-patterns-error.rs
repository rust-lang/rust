//! A parameter that is not a plain name, together with an autodiff error. Expansion builds the
//! generated function from the recovered signature, and that used to ICE (see #150328).

#![feature(autodiff)]

use std::autodiff::autodiff_reverse;

#[autodiff_reverse(df, Duplicated, Active)]
fn f(_: f64) -> f64 {
    //~^ ERROR Duplicated can not be used for this type
    1.0
}

fn main() {}
