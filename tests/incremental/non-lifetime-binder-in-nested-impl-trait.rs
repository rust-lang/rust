//@ revisions: cfail
//@ should-ice
//@ compile-flags: --edition=2021
//@ error-pattern: assertion failed

#![feature(non_lifetime_binders)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]

trait Trait<T: ?Sized> {
    type Assoc<'a> = i32;
}

fn produce() -> impl for<T> Trait<(), Assoc = impl Trait<T>> {
    16
}

fn main() {}
