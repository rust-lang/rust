#![feature(prelude_import)]
#![no_std]
#![feature(autodiff)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:autodiff_forward.pp

// Test that forward mode ad macros are expanded correctly.

#[rustc_autodiff]
#[inline(never)]
pub fn f1(x: &[f64], y: f64) -> f64 {



    // Not the most interesting derivative, but who are we to judge

    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, Dual, Const, Dual,)]
#[inline(never)]
pub fn df1(x: &[f64], bx: &[f64], y: f64) -> (f64, f64) {
    unsafe { asm!("NOP"); };
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
    unsafe { asm!("NOP"); };
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
    unsafe { asm!("NOP"); };
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
    unsafe { asm!("NOP"); };
    ::core::hint::black_box(f4());
    ::core::hint::black_box(());
}
fn main() {}
