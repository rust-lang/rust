//! Previously, `Foo` was accepted as it has no generics but `Bar` was not.
//@ check-pass

macro_rules! unit {
    () => {
        ()
    };
}

#[derive(Clone)]
struct Foo {
    field: unit!(),
}

#[derive(Clone)]
struct Bar<T> {
    field: unit!(),
    other: T,
}

fn main() {}
