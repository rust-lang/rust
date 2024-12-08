//@ needs-enzyme

#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_forward.pp

// Test that forward mode ad macros are expanded correctly.

use std::autodiff::autodiff;

#[autodiff(df1, Forward, Dual, Const, Dual)]
pub fn f1(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

#[autodiff(df2, Forward, Dual, Const, Const)]
pub fn f2(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

#[autodiff(df3, ForwardFirst, Dual, Const, Const)]
pub fn f3(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

// Not the most interesting derivative, but who are we to judge
#[autodiff(df4, Forward)]
pub fn f4() {}

// We want to be sure that the same function can be differentiated in different ways
#[autodiff(df5_rev, Reverse, Duplicated, Const, Active)]
#[autodiff(df5_x, Forward, Dual, Const, Const)]
#[autodiff(df5_y, Forward, Const, Dual, Const)]
pub fn f5(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

fn main() {}
