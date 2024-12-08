#![deny(clippy::use_self)]

pub struct Foo;

impl Foo {
    pub fn bar() -> Foo {
        Foo
    }
}

fn main() {}
