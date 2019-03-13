#![feature(trait_alias)]

mod inner {
    pub trait Foo {
        fn foo(&self);
    }

    pub struct Qux;

    impl Foo for Qux {
        fn foo(&self) {}
    }

    pub trait Bar = Foo;
}

// Import only the alias, not the `Foo` trait.
use inner::{Bar, Qux};

fn main() {
    let q = Qux;
    q.foo();
}
