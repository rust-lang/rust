#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, UnalignedField, field_of};

pub struct MyStruct(());

impl Drop for field_of!(MyStruct, 0) {
    //~^ ERROR: the `Drop` trait may only be implemented for local structs, enums, and unions [E0120]
    fn drop(&mut self) {}
}

fn main() {}
