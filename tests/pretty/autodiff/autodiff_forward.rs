//@ needs-enzyme

#![feature(autodiff)]
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_forward.pp

// Test that forward mode ad macros are expanded correctly.

use std::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(df1, Dual, Const, Dual)]
pub fn f1(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

#[autodiff_forward(df2, Dual, Const, Const)]
pub fn f2(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

#[autodiff_forward(df3, Dual, Const, Const)]
pub fn f3(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

// Not the most interesting derivative, but who are we to judge
#[autodiff_forward(df4)]
pub fn f4() {}

// We want to be sure that the same function can be differentiated in different ways
#[autodiff_reverse(df5_rev, Duplicated, Const, Active)]
#[autodiff_forward(df5_x, Dual, Const, Const)]
#[autodiff_forward(df5_y, Const, Dual, Const)]
pub fn f5(x: &[f64], y: f64) -> f64 {
    unimplemented!()
}

struct DoesNotImplDefault;
#[autodiff_forward(df6, Const)]
pub fn f6() -> DoesNotImplDefault {
    unimplemented!()
}

// Make sure, that we add the None for the default return.
#[autodiff_forward(df7, Const)]
pub fn f7(x: f32) -> () {}

#[autodiff_forward(f8_1, Dual, DualOnly)]
#[autodiff_forward(f8_2, 4, Dual, DualOnly)]
#[autodiff_forward(f8_3, 4, Dual, Dual)]
#[no_mangle]
fn f8(x: &f32) -> f32 {
    unimplemented!()
}

// We want to make sure that we can use the macro for functions defined inside of functions
pub fn f9() {
    #[autodiff_forward(d_inner_1, Dual, DualOnly)]
    #[autodiff_forward(d_inner_2, Dual, Dual)]
    fn inner(x: f32) -> f32 {
        x * x
    }
}

// Make sure we can handle generics
#[autodiff_reverse(d_square, Duplicated, Active)]
pub fn f10<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T {
    *x * *x
}

fn main() {}
