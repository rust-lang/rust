//! Ensure we allow tuples behind `adt_const_params`
//@check-pass
#![feature(min_adt_const_params)]

#[allow(dead_code)]
fn foo<const N: (i32, u32, i16)>() {}

fn main() {}
