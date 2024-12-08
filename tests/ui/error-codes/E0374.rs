#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> { //~ ERROR `T` is never used
    a: i32,
}

impl<T, U> CoerceUnsized<Foo<U>> for Foo<T> //~ ERROR E0374
    where T: CoerceUnsized<U> {}

fn main() {}
