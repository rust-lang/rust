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

use std::autodiff::{autodiff_forward, autodiff_reverse};

#[rustc_autodiff]
#[inline(never)]
pub fn f1(x: &[f64], y: f64) -> f64 {



    // Not the most interesting derivative, but who are we to judge

    // We want to be sure that the same function can be differentiated in different ways


    // Make sure, that we add the None for the default return.


    // We want to make sure that we can use the macro for functions defined inside of functions

    // Make sure we can handle generics

    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Dual, Const, Dual)]
#[inline(never)]
pub fn df1(x: &[f64], bx_0: &[f64], y: f64) -> (f64, f64) {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f1(x, y));
    ::core::hint::black_box((bx_0,));
    ::core::hint::black_box(<(f64, f64)>::default())
}
#[rustc_autodiff]
#[inline(never)]
pub fn f2(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Dual, Const, Const)]
#[inline(never)]
pub fn df2(x: &[f64], bx_0: &[f64], y: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f2(x, y));
    ::core::hint::black_box((bx_0,));
    ::core::hint::black_box(f2(x, y))
}
#[rustc_autodiff]
#[inline(never)]
pub fn f3(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Dual, Const, Const)]
#[inline(never)]
pub fn df3(x: &[f64], bx_0: &[f64], y: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f3(x, y));
    ::core::hint::black_box((bx_0,));
    ::core::hint::black_box(f3(x, y))
}
#[rustc_autodiff]
#[inline(never)]
pub fn f4() {}
#[rustc_autodiff(Forward, 1, None)]
#[inline(never)]
pub fn df4() -> () {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f4());
    ::core::hint::black_box(());
}
#[rustc_autodiff]
#[inline(never)]
pub fn f5(x: &[f64], y: f64) -> f64 {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Const, Dual, Const)]
#[inline(never)]
pub fn df5_y(x: &[f64], y: f64, by_0: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((by_0,));
    ::core::hint::black_box(f5(x, y))
}
#[rustc_autodiff(Forward, 1, Dual, Const, Const)]
#[inline(never)]
pub fn df5_x(x: &[f64], bx_0: &[f64], y: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((bx_0,));
    ::core::hint::black_box(f5(x, y))
}
#[rustc_autodiff(Reverse, 1, Duplicated, Const, Active)]
#[inline(never)]
pub fn df5_rev(x: &[f64], dx_0: &mut [f64], y: f64, dret: f64) -> f64 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f5(x, y));
    ::core::hint::black_box((dx_0, dret));
    ::core::hint::black_box(f5(x, y))
}
struct DoesNotImplDefault;
#[rustc_autodiff]
#[inline(never)]
pub fn f6() -> DoesNotImplDefault {
    ::core::panicking::panic("not implemented")
}
#[rustc_autodiff(Forward, 1, Const)]
#[inline(never)]
pub fn df6() -> DoesNotImplDefault {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f6());
    ::core::hint::black_box(());
    ::core::hint::black_box(f6())
}
#[rustc_autodiff]
#[inline(never)]
pub fn f7(x: f32) -> () {}
#[rustc_autodiff(Forward, 1, Const, None)]
#[inline(never)]
pub fn df7(x: f32) -> () {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f7(x));
    ::core::hint::black_box(());
}
#[no_mangle]
#[rustc_autodiff]
#[inline(never)]
fn f8(x: &f32) -> f32 { ::core::panicking::panic("not implemented") }
#[rustc_autodiff(Forward, 4, Dual, Dual)]
#[inline(never)]
fn f8_3(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32, bx_3: &f32)
    -> [f32; 5usize] {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f8(x));
    ::core::hint::black_box((bx_0, bx_1, bx_2, bx_3));
    ::core::hint::black_box(<[f32; 5usize]>::default())
}
#[rustc_autodiff(Forward, 4, Dual, DualOnly)]
#[inline(never)]
fn f8_2(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32, bx_3: &f32)
    -> [f32; 4usize] {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f8(x));
    ::core::hint::black_box((bx_0, bx_1, bx_2, bx_3));
    ::core::hint::black_box(<[f32; 4usize]>::default())
}
#[rustc_autodiff(Forward, 1, Dual, DualOnly)]
#[inline(never)]
fn f8_1(x: &f32, bx_0: &f32) -> f32 {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f8(x));
    ::core::hint::black_box((bx_0,));
    ::core::hint::black_box(<f32>::default())
}
pub fn f9() {
    #[rustc_autodiff]
    #[inline(never)]
    fn inner(x: f32) -> f32 { x * x }
    #[rustc_autodiff(Forward, 1, Dual, Dual)]
    #[inline(never)]
    fn d_inner_2(x: f32, bx_0: f32) -> (f32, f32) {
        unsafe { asm!("NOP", options(pure, nomem)); };
        ::core::hint::black_box(inner(x));
        ::core::hint::black_box((bx_0,));
        ::core::hint::black_box(<(f32, f32)>::default())
    }
    #[rustc_autodiff(Forward, 1, Dual, DualOnly)]
    #[inline(never)]
    fn d_inner_1(x: f32, bx_0: f32) -> f32 {
        unsafe { asm!("NOP", options(pure, nomem)); };
        ::core::hint::black_box(inner(x));
        ::core::hint::black_box((bx_0,));
        ::core::hint::black_box(<f32>::default())
    }
}
#[rustc_autodiff]
#[inline(never)]
pub fn f10<T: std::ops::Mul<Output = T> + Copy>(x: &T) -> T { *x * *x }
#[rustc_autodiff(Reverse, 1, Duplicated, Active)]
#[inline(never)]
pub fn d_square<T: std::ops::Mul<Output = T> +
    Copy>(x: &T, dx_0: &mut T, dret: T) -> T {
    unsafe { asm!("NOP", options(pure, nomem)); };
    ::core::hint::black_box(f10::<T>(x));
    ::core::hint::black_box((dx_0, dret));
    ::core::hint::black_box(f10::<T>(x))
}
fn main() {}
