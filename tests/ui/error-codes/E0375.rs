#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized, U: ?Sized> {
    a: i32,
    b: T, //~ ERROR E0277
    c: U,
}

impl<T, U> CoerceUnsized<Foo<U, T>> for Foo<T, U> {}
//~^ ERROR E0375

fn main() {}
