#![feature(prelude_import)]
#![no_std]
//@ needs-enzyme

#![feature(autodiff)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_reverse.pp

// Test that reverse mode ad macros are expanded correctly.

use std::autodiff::autodiff;

#[rustc_autodiff]
#[inline(never)]
pub fn f1(x: &[f64], y: f64) -> f64 {

    // Not the most interesting derivative, but who are we to judge


    // What happens if we already have Reverse in type (enum variant decl) and value (enum variant
    // constructor) namespace? > It's expected to work normally.


    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Reverse, 1, Duplicated, Const, Active)]
#[inline(never)]
pub fn df1(x: &[f64], dx_0: &mut [f64], y: f64, dret: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f1(x, y));
    ::core::hint::black_box((dx_0, dret));
    ::core::hint::black_box(f1(x, y))
}
#[rustc_autodiff]
#[inline(never)]
pub fn f2() {}
#[rustc_autodiff(Reverse, 1, None)]
#[inline(never)]
pub fn df2() {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f2());
    ::core::hint::black_box(());
}
#[rustc_autodiff]
#[inline(never)]
pub fn f3(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Reverse, 1, Duplicated, Const, Active)]
#[inline(never)]
pub fn df3(x: &[f64], dx_0: &mut [f64], y: f64, dret: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f3(x, y));
    ::core::hint::black_box((dx_0, dret));
    ::core::hint::black_box(f3(x, y))
}
enum Foo { Reverse, }
use Foo::Reverse;
#[rustc_autodiff]
#[inline(never)]
pub fn f4(x: f32) { ::core::panicking::panic("not implemented") }
#[rustc_autodiff(Reverse, 1, Const, None)]
#[inline(never)]
pub fn df4(x: f32) {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f4(x));
    ::core::hint::black_box(());
}
#[rustc_autodiff]
#[inline(never)]
pub fn f5(x: *const f32, y: &f32) {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Reverse, 1, DuplicatedOnly, Duplicated, None)]
#[inline(never)]
pub unsafe fn df5(x: *const f32, dx_0: *mut f32, y: &f32, dy_0: &mut f32) {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((dx_0, dy_0));
}
fn main() {}
