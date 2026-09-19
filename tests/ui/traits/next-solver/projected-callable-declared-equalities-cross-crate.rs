//@ run-pass
//@ compile-flags: -Znext-solver=globally
//@ aux-build: declared_equality_aux.rs

extern crate declared_equality_aux;

use declared_equality_aux::{Carrier, Family, Identity, identity};

fn invoke<'a, C: Carrier<Assoc = T>, T: Family>(value: T::View<'a>) -> T::View<'a> {
    let f: for<'b> fn(T::View<'b>) -> T::View<'b> = identity::<C, T>;
    f(value)
}

struct Borrowed;

impl Family for Borrowed {
    type View<'a> = &'a u32;
}

impl Identity for Borrowed {
    type Output = Self;
}

impl Carrier for Borrowed {
    type Assoc = Self;
}

fn main() {
    let value = 43;
    assert_eq!(*invoke::<Borrowed, Borrowed>(&value), 43);
}
