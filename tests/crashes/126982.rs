//@ known-bug: rust-lang/rust#126982

#![feature(coerce_unsized)]
use std::ops::CoerceUnsized;

struct Foo<T: ?Sized> {
    a: T,
}

impl<T, U> CoerceUnsized<U> for Foo<T> {}

union U {
    a: usize,
}

const C: U = Foo { a: 10 };

fn main() {}
