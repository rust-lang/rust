#![feature(autodiff)]
extern crate simple_dep;
use std::autodiff::*;

#[inline(never)]
pub fn f2(x: f64) -> f64 {
    x.sin()
}

#[autodiff_forward(df1_lib, Dual, Dual)]
pub fn _f1(x: f64) -> f64 {
    simple_dep::f(x, x) * f2(x)
}
