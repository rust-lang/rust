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
    assert_field::<field_of!(MyStruct, 0)>(); //[next]~ ERROR: the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time [E0277]
    //[old]~^ ERROR: the trait bound `field_of!(MyStruct, 0): Field` is not satisfied [E0277]

    // FIXME(FRTs): improve this error message, point to the `dyn Trait` span.
    assert_field::<field_of!(MyStruct, 1)>(); //[next]~ ERROR: the size for values of type `(dyn Trait + 'static)` cannot be known at compilation time [E0277]
    //[old]~^ ERROR: the trait bound `field_of!(MyStruct, 1): Field` is not satisfied [E0277]
}
