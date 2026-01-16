//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

pub struct MyStruct(());

impl Drop for field_of!(MyStruct, 0) {
    //~^ ERROR: the `Drop` trait may only be implemented for local structs, enums, and unions [E0120]
    fn drop(&mut self) {}
}

unsafe impl Send for field_of!(MyStruct, 0) {}
//~^ ERROR: impls of auto traits for field representing types not supported

#[repr(packed)]
pub struct MyStruct2(usize);

unsafe impl Field for field_of!(MyStruct2, 0) {
    //~^ ERROR: explicit impls for the `Field` trait are not permitted [E0322]
    type Base = MyStruct2;
    type Type = usize;
    const OFFSET: usize = 0;
}

pub struct MyField;

unsafe impl Field for MyField {
    //~^ ERROR: explicit impls for the `Field` trait are not permitted [E0322]
    type Base = ();
    type Type = ();
    const OFFSET: usize = 0;
}

fn main() {}
