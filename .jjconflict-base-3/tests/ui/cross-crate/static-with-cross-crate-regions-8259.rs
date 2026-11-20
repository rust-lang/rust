// https://github.com/rust-lang/rust/issues/8259
//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

//@ aux-build:aux-8259.rs

extern crate aux_8259 as other;
static a: other::Foo<'static> = other::Foo::A;

pub fn main() {}
