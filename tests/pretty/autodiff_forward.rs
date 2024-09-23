#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_forward.pp

// Test that forward mode ad macros are expanded correctly.

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

fn main() {}
