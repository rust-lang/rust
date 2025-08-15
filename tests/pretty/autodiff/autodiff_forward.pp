#![feature(prelude_import)]
#![no_std]
//@ needs-enzyme

#![feature(autodiff)]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_forward.pp

// Test that forward mode ad macros are expanded correctly.

use std::autodiff::{autodiff_forward, autodiff_reverse};

#[rustc_autodiff]
pub fn f1(x: &[f64], y: f64) -> f64 {



    // Not the most interesting derivative, but who are we to judge

    // We want to be sure that the same function can be differentiated in different ways


    // Make sure, that we add the None for the default return.


    // We want to make sure that we can use the macro for functions defined inside of functions

    // Make sure we can handle generics

    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Dual, Const, Dual)]
pub fn df1(x: &[f64], bx_0: &[f64], y: f64) -> (f64, f64) {
    ::core::intrinsics::autodiff(f1::<>, df1::<>, (x, bx_0, y))
}
#[rustc_autodiff]
pub fn f2(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Dual, Const, Const)]
pub fn df2(x: &[f64], bx_0: &[f64], y: f64) -> f64 {
    ::core::intrinsics::autodiff(f2::<>, df2::<>, (x, bx_0, y))
}
#[rustc_autodiff]
pub fn f3(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Dual, Const, Const)]
pub fn df3(x: &[f64], bx_0: &[f64], y: f64) -> f64 {
    ::core::intrinsics::autodiff(f3::<>, df3::<>, (x, bx_0, y))
}
#[rustc_autodiff]
pub fn f4() {}
#[rustc_autodiff(Forward, 1, None)]
pub fn df4() -> () { ::core::intrinsics::autodiff(f4::<>, df4::<>, ()) }
#[rustc_autodiff]
pub fn f5(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Const, Dual, Const)]
pub fn df5_y(x: &[f64], y: f64, by_0: f64) -> f64 {
    ::core::intrinsics::autodiff(f5::<>, df5_y::<>, (x, y, by_0))
}
#[rustc_autodiff(Forward, 1, Dual, Const, Const)]
pub fn df5_x(x: &[f64], bx_0: &[f64], y: f64) -> f64 {
    ::core::intrinsics::autodiff(f5::<>, df5_x::<>, (x, bx_0, y))
}
#[rustc_autodiff(Reverse, 1, Duplicated, Const, Active)]
pub fn df5_rev(x: &[f64], dx_0: &mut [f64], y: f64, dret: f64) -> f64 {
    ::core::intrinsics::autodiff(f5::<>, df5_rev::<>, (x, dx_0, y, dret))
}
struct DoesNotImplDefault;
#[rustc_autodiff]
pub fn f6() -> DoesNotImplDefault {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Const)]
pub fn df6() -> DoesNotImplDefault {
    ::core::intrinsics::autodiff(f6::<>, df6::<>, ())
}
#[rustc_autodiff]
pub fn f7(x: f32) -> () {}
#[rustc_autodiff(Forward, 1, Const, None)]
pub fn df7(x: f32) -> () {
    ::core::intrinsics::autodiff(f7::<>, df7::<>, (x,))
}
#[no_mangle]
#[rustc_autodiff]
fn f8(x: &f32) -> f32 { ::core::panicking::panic("not implemented") }
#[rustc_autodiff(Forward, 4, Dual, Dual)]
fn f8_3(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32, bx_3: &f32)
    -> [f32; 5usize] {
    ::core::intrinsics::autodiff(f8::<>, f8_3::<>,
        (x, bx_0, bx_1, bx_2, bx_3))
}
#[rustc_autodiff(Forward, 4, Dual, DualOnly)]
fn f8_2(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32, bx_3: &f32)
    -> [f32; 4usize] {
    ::core::intrinsics::autodiff(f8::<>, f8_2::<>,
        (x, bx_0, bx_1, bx_2, bx_3))
}
#[rustc_autodiff(Forward, 1, Dual, DualOnly)]
fn f8_1(x: &f32, bx_0: &f32) -> f32 {
    ::core::intrinsics::autodiff(f8::<>, f8_1::<>, (x, bx_0))
}
pub fn f9() {
    #[rustc_autodiff]
    fn inner(x: f32) -> f32 { x * x }
    #[rustc_autodiff(Forward, 1, Dual, Dual)]
    fn d_inner_2(x: f32, bx_0: f32) -> (f32, f32) {
        ::core::intrinsics::autodiff(inner::<>, d_inner_2::<>, (x, bx_0))
    }
    #[rustc_autodiff(Forward, 1, Dual, DualOnly)]
    fn d_inner_1(x: f32, bx_0: f32) -> f32 {
        ::core::intrinsics::autodiff(inner::<>, d_inner_1::<>, (x, bx_0))
    }
}
#[rustc_autodiff]
pub fn f10<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T { *x * *x }
#[rustc_autodiff(Reverse, 1, Duplicated, Active)]
pub fn d_square<T: std::ops::Mul<Output = T> +
    Copy>(x: &T, dx_0: &mut T, dret: T) -> T {
    ::core::intrinsics::autodiff(f10::<T>, d_square::<T>, (x, dx_0, dret))
}
fn main() {}
