//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

#[repr(packed)]
pub struct MyStruct(usize);

fn assert_field<F: Field>() {}

fn main() {
    // FIXME(FRTs): improve this error message, point to the `repr(packed)` span.
    assert_field::<field_of!(MyStruct, 0)>();
    //~^ ERROR: the trait bound `field_of!(MyStruct, 0): std::field::Field` is not satisfied [E0277]
}
