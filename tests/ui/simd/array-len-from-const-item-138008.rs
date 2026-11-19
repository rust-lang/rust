//! Regression test for <https://github.com/rust-lang/rust/issues/138008>.
//@ build-pass
#![feature(repr_simd)]
#![crate_type = "lib"]

const C: usize = 16;

#[repr(simd)]
pub struct Foo([u8; C]);

pub unsafe fn foo(a: Foo) {}
