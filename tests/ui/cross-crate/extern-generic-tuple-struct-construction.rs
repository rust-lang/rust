//@ run-pass
//@ aux-build:extern-generic-tuple-struct-construction.rs
//! Regression test for https://github.com/rust-lang/rust/issues/4545

extern crate extern_generic_tuple_struct_construction as somelib;
pub fn main() { somelib::mk::<isize>(); }
