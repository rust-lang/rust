//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

pub trait Trait {}

pub struct MyStruct(usize, dyn Trait);

fn assert_field<F: Field>() {}

fn main() {
    // FIXME(FRTs): this requires relaxing the `Base: ?Sized` bound in the
    // `Field` trait & compiler changes.
    assert_field::<field_of!(MyStruct, 0)>();
    //~^ ERROR: the trait bound `field_of!(MyStruct, 0): std::field::Field` is not satisfied [E0277]

    // FIXME(FRTs): improve this error message, point to the `dyn Trait` span.
    assert_field::<field_of!(MyStruct, 1)>();
    //~^ ERROR: the trait bound `field_of!(MyStruct, 1): std::field::Field` is not satisfied [E0277]
}
