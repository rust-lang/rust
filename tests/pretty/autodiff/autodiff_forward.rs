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

#[autodiff(df3, Forward, Dual, Const, Const)]
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

struct DoesNotImplDefault;
#[autodiff(df6, Forward, Const)]
pub fn f6() -> DoesNotImplDefault {
    unimplemented!()
}

// Make sure, that we add the None for the default return.
#[autodiff(df7, Forward, Const)]
pub fn f7(x: f32) -> () {}

#[autodiff(f8_1, Forward, Dual, DualOnly)]
#[autodiff(f8_2, Forward, 4, Dual, DualOnly)]
#[autodiff(f8_3, Forward, 4, Dual, Dual)]
#[no_mangle]
fn f8(x: &f32) -> f32 {
    unimplemented!()
}

// We want to make sure that we can use the macro for functions defined inside of functions
pub fn f9() {
    #[autodiff(d_inner_1, Forward, Dual, DualOnly)]
    #[autodiff(d_inner_2, Forward, Dual, Dual)]
    fn inner(x: f32) -> f32 {
        x * x
    }
}

// Make sure we can handle generics
#[autodiff(d_square, Reverse, Duplicated, Active)]
pub fn f10<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T {
    *x * *x
}

fn main() {}
