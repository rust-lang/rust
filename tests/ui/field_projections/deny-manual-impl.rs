#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, UnalignedField, field_of};

#[repr(packed)]
pub struct MyStruct(usize);

unsafe impl UnalignedField for MyStruct {
    //~^ ERROR: the `UnalignedField` trait may not be implemented manually [E0806]
    type Base = ();
    type Type = ();
    const OFFSET: usize = 0;
}

unsafe impl Field for field_of!(MyStruct, 0) {} //~ ERROR: the `Field` trait may not be implemented manually [E0806]

fn main() {}
