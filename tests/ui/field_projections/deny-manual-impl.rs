#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, UnalignedField, field_of};

#[repr(packed)]
pub struct MyStruct(usize);

unsafe impl UnalignedField for MyStruct {
    //~^ ERROR: explicit impls for the `UnalignedField` trait are not permitted [E0322]
    type Base = ();
    type Type = ();
    const OFFSET: usize = 0;
}

unsafe impl Field for field_of!(MyStruct, 0) {} //~ ERROR: explicit impls for the `Field` trait are not permitted [E0322]

fn main() {}
