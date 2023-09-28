// build-pass

#![feature(adt_const_params)]
//~^ WARN the feature `adt_const_params` is incomplete

pub trait Identity {
    type Identity;
}

impl<T> Identity for T {
    type Identity = Self;
}

pub fn foo<const X: <i32 as Identity>::Identity>() {
    assert!(X == 12);
}

fn main() {
    foo::<12>();
}
