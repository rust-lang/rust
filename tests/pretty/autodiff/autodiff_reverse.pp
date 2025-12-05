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
//@ pp-exact:autodiff_reverse.pp

// Test that reverse mode ad macros are expanded correctly.

use std::autodiff::autodiff_reverse;

#[rustc_autodiff]
pub fn f1(x: &[f64], y: f64) -> f64 {

    // Not the most interesting derivative, but who are we to judge


    // What happens if we already have Reverse in type (enum variant decl) and value (enum variant
    // constructor) namespace? > It's expected to work normally.


    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Reverse, 1, Duplicated, Const, Active)]
pub fn df1(x: &[f64], dx_0: &mut [f64], y: f64, dret: f64) -> f64 {
    ::core::intrinsics::autodiff(f1::<>, df1::<>, (x, dx_0, y, dret))
}
#[rustc_autodiff]
pub fn f2() {}
#[rustc_autodiff(Reverse, 1, None)]
pub fn df2() { ::core::intrinsics::autodiff(f2::<>, df2::<>, ()) }
#[rustc_autodiff]
pub fn f3(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Reverse, 1, Duplicated, Const, Active)]
pub fn df3(x: &[f64], dx_0: &mut [f64], y: f64, dret: f64) -> f64 {
    ::core::intrinsics::autodiff(f3::<>, df3::<>, (x, dx_0, y, dret))
}
enum Foo { Reverse, }
use Foo::Reverse;
#[rustc_autodiff]
pub fn f4(x: f32) { ::core::panicking::panic("not implemented") }
#[rustc_autodiff(Reverse, 1, Const, None)]
pub fn df4(x: f32) { ::core::intrinsics::autodiff(f4::<>, df4::<>, (x,)) }
#[rustc_autodiff]
pub fn f5(x: *const f32, y: &f32) {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Reverse, 1, DuplicatedOnly, Duplicated, None)]
pub unsafe fn df5(x: *const f32, dx_0: *mut f32, y: &f32, dy_0: &mut f32) {
    ::core::intrinsics::autodiff(f5::<>, df5::<>, (x, dx_0, y, dy_0))
}
fn main() {}
