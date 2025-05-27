#![feature(rustc_attrs)]
#![feature(prelude_import)]
#![feature(panic_internals)]
#![no_std]
//@ needs-enzyme
#![feature(autodiff)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:batching.pp


// Test that forward mode ad macros are expanded correctly.
use std::arch::asm;
use std::autodiff::autodiff;

#[no_mangle]
#[rustc_autodiff]
#[inline(never)]
fn square(x: &f32) -> f32 {
    x * x
}
#[rustc_autodiff(Forward, 4, Dual, Dual)]
#[inline(never)]
fn d_square1(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32, bx_3: &f32) -> [f32; 5usize] {
    unsafe {
        asm!("NOP", options(nomem));
    };
    ::core::hint::black_box(square(x));
    ::core::hint::black_box((bx_0, bx_1, bx_2, bx_3));
    ::core::hint::black_box(<[f32; 5usize]>::default())
}
#[rustc_autodiff(Forward, 4, Dual, DualOnly)]
#[inline(never)]
fn d_square2(x: &f32, bx_0: &f32, bx_1: &f32, bx_2: &f32, bx_3: &f32) -> [f32; 4usize] {
    unsafe {
        asm!("NOP", options(nomem));
    };
    ::core::hint::black_box(square(x));
    ::core::hint::black_box((bx_0, bx_1, bx_2, bx_3));
    ::core::hint::black_box(<[f32; 4usize]>::default())
}


fn main() {}
