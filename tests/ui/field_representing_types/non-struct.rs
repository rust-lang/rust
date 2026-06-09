//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};

pub union MyUnion {
    field: u32,
    other: i32,
}

pub enum MyEnum {
    A { a: i32, b: i64 },
    B { x: i64, y: i32 },
}

fn assert_field<F: Field>() {}

fn main() {
    // FIXME(FRTs): improve this error message, point to the `union`.
    assert_field::<field_of!(MyUnion, field)>();
    //~^ ERROR: the trait bound `field_of!(MyUnion, field): std::field::Field` is not satisfied [E0277]
    // FIXME(FRTs): improve this error message, point to the `enum`.
    assert_field::<field_of!(MyEnum, A.a)>();
    //~^ ERROR: the trait bound `field_of!(MyEnum, A.a): std::field::Field` is not satisfied [E0277]
}
