//@ check-pass
//@ compile-flags: --crate-type=lib

#[allow(private_bounds)]
pub trait Foo: FooImpl {}

trait FooImpl {}
