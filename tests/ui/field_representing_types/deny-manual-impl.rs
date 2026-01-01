//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::Field;

#[repr(packed)]
pub struct MyStruct(usize);

unsafe impl Field for MyStruct {
    //~^ ERROR: explicit impls for the `Field` trait are not permitted [E0322]
    type Base = ();
    type Type = ();
    const OFFSET: usize = 0;
}

fn main() {}
