//@ compile-flags: -Zautodiff=Enable,PrintModAfter -Zautodiff=NoPostopt -C opt-level=3  -Clto=fat
//@ no-prefer-dynamic
//@ needs-enzyme
//
#![feature(rustc_attrs)]
#![feature(prelude_import)]
#![feature(panic_internals)]
#![feature(portable_simd)]
#![no_std]
//@ needs-enzyme
#![feature(autodiff)]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
use std::simd::Simd;
//@ pretty-mode:expanded
//@ pretty-compare-only
//@ pp-exact:batching.pp

// CHECK: __enzyme

// Test that forward mode ad macros are expanded correctly.
use std::arch::asm;
use std::autodiff::autodiff;

// Generated from:
//// ```
/// #[batching(d_square2, 4, Buffer, Buffer)]
/// fn square(x: f32) -> f32 {
///    x * x
/// }

#[no_mangle]
#[rustc_autodiff]
#[inline(never)]
fn square(x: &[f32]) -> f32 {
    x[0] * x[0]
}
#[rustc_autodiff(Batch, 4, Buffer, Buffer)]
#[no_mangle]
#[inline(never)]
fn d_square2(x: &[f32]) -> Simd<f32,4> {
    unsafe {
        asm!("NOP", options(nomem));
    };
    ::core::hint::black_box(square(x));
    ::core::hint::black_box(());
    ::core::hint::black_box(Default::default())
}
//#[rustc_autodiff(Batch, 4, Batching, Batching)]
//#[no_mangle]
//#[inline(never)]
//fn d_square2(x: Simd<f32,4>) -> Simd<f32,4> {
//    unsafe {
//        asm!("NOP", options(nomem));
//    };
//    ::core::hint::black_box(square(x[0]));
//    ::core::hint::black_box(());
//    ::core::hint::black_box(Default::default())
//}
fn main() {}
