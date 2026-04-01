//@ check-pass
#![feature(adt_const_params)]

// Regression test for #116308

pub trait Identity {
    type Identity;
}

impl<T> Identity for T {
    type Identity = Self;
}

pub fn foo<const X: <i32 as Identity>::Identity>() {}

fn main() {
    foo::<12>();
}
