// Tests that we correctly handle the instantiated
// inference variable being completely unconstrained.
//
// check-pass
#![feature(type_alias_impl_trait)]

type Foo = impl Copy;

enum Wrapper<T> {
    First(T),
    Second
}

fn _make_iter() -> Foo {
    true
}

fn _produce() -> Wrapper<Foo> {
    Wrapper::Second
}

fn main() {}
