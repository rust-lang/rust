//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]
#![allow(refining_impl_trait)]

#[const_trait]
pub trait Foo {
    (const) fn method(self) -> impl ~const Bar;
}

#[const_trait]
pub trait Bar {}

struct A<T>(T);
impl<T> const Foo for A<T> where A<T>: ~const Bar {
    (const) fn method(self) -> impl ~const Bar {
        self
    }
}

fn main() {}
