// Regression test for #60976: ICE (with <=1.36.0) when another file had `use <primitive_type>;`.
//@ check-pass
//@ aux-build:extern-use-primitive-type-lib.rs

extern crate extern_use_primitive_type_lib;

fn main() {}
