//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::field::{Field, field_of};
use std::fmt::Debug;

pub struct MyDST {
    count: usize,
    last: dyn Debug,
}

pub struct Generic<T: ?Sized> {
    count: usize,
    last: T,
}

fn generic<T: ?Sized>() {
    impls_field::<field_of!(Generic<T>, count)>();
    //~^ ERROR: the trait bound `field_of!(Generic<T>, count): std::field::Field` is not satisfied [E0277]
    impls_field::<field_of!(Generic<T>, last)>();
    //~^ ERROR: the trait bound `field_of!(Generic<T>, last): std::field::Field` is not satisfied [E0277]
}

fn ok<T>() {
    impls_field::<field_of!(Generic<T>, count)>();
    impls_field::<field_of!(Generic<T>, last)>();
}

fn main() {
    impls_field::<field_of!(MyDST, count)>();
    //~^ ERROR: the trait bound `field_of!(MyDST, count): std::field::Field` is not satisfied [E0277]
    impls_field::<field_of!(MyDST, last)>();
    //~^ ERROR: the trait bound `field_of!(MyDST, last): std::field::Field` is not satisfied [E0277]
}

fn impls_field<F: Field>() {}
