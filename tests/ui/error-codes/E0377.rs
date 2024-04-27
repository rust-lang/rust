#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

pub struct Foo<T: ?Sized> {
    field_with_unsized_type: T,
}

pub struct Bar<T: ?Sized> {
    field_with_unsized_type: T,
}

impl<T, U> CoerceUnsized<Bar<U>> for Foo<T> where T: CoerceUnsized<U> {} //~ ERROR E0377

fn main() {}
