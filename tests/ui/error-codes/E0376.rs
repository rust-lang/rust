#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: T,
}

impl<T, U> CoerceUnsized<U> for Foo<T> {} //~ ERROR E0376

fn main() {}
