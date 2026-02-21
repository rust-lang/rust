// Regression test for #138008.
//@ build-fail
//@ compile-flags: --crate-type=lib -Copt-level=0

#![feature(repr_simd)]

const C: usize = 16;

#[repr(simd)]
pub struct Foo([u8; C]);

pub unsafe fn foo(a: Foo) {} //~ ERROR the type `Foo` has an unknown layout
