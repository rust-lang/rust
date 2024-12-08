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
//@ pp-exact:autodiff_forward.pp

// Test that forward mode ad macros are expanded correctly.

use std::autodiff::autodiff;

#[rustc_autodiff]
#[inline(never)]
pub fn f1(x: &[f64], y: f64) -> f64 {



    // Not the most interesting derivative, but who are we to judge

    // We want to be sure that the same function can be differentiated in different ways

    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, Dual, Const, Dual,)]
#[inline(never)]
pub fn df1(x: &[f64], bx: &[f64], y: f64) -> (f64, f64) {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f1(x, y));
    ::core::hint::black_box((bx,));
    ::core::hint::black_box((f1(x, y), f64::default()))
}
#[rustc_autodiff]
#[inline(never)]
pub fn f2(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, Dual, Const, Const,)]
#[inline(never)]
pub fn df2(x: &[f64], bx: &[f64], y: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f2(x, y));
    ::core::hint::black_box((bx,));
    ::core::hint::black_box(f2(x, y))
}
#[rustc_autodiff]
#[inline(never)]
pub fn f3(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(ForwardFirst, Dual, Const, Const,)]
#[inline(never)]
pub fn df3(x: &[f64], bx: &[f64], y: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f3(x, y));
    ::core::hint::black_box((bx,));
    ::core::hint::black_box(f3(x, y))
}
#[rustc_autodiff]
#[inline(never)]
pub fn f4() {}
#[rustc_autodiff(Forward, None)]
#[inline(never)]
pub fn df4() {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f4());
    ::core::hint::black_box(());
}
#[rustc_autodiff]
#[inline(never)]
#[rustc_autodiff]
#[inline(never)]
#[rustc_autodiff]
#[inline(never)]
pub fn f5(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, Const, Dual, Const,)]
#[inline(never)]
pub fn df5_y(x: &[f64], y: f64, by: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((by,));
    ::core::hint::black_box(f5(x, y))
}
#[rustc_autodiff(Forward, Dual, Const, Const,)]
#[inline(never)]
pub fn df5_x(x: &[f64], bx: &[f64], y: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((bx,));
    ::core::hint::black_box(f5(x, y))
}
#[rustc_autodiff(Reverse, Duplicated, Const, Active,)]
#[inline(never)]
pub fn df5_rev(x: &[f64], dx: &mut [f64], y: f64, dret: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((dx, dret));
    ::core::hint::black_box(f5(x, y))
}
fn main() {}
