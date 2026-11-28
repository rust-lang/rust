// Regression test for <https://github.com/rust-lang/rust/issues/166089>.

//@ build-pass
//@ compile-flags: -Copt-level=2

#![feature(repr_simd)]
#![crate_type = "lib"]
#![allow(invalid_value)]

use std::mem::MaybeUninit;

#[repr(simd)]
#[derive(Copy, Clone)]
pub struct U64x1([u64; 1]);

#[inline(never)]
fn take(_: U64x1) {}

pub fn f() {
    let v: U64x1 = unsafe { MaybeUninit::uninit().assume_init() };
    take(v);
}
