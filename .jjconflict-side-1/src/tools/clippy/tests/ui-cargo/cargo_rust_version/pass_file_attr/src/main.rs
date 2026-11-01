#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.13.0"]
#![deny(clippy::use_self)]

pub struct Foo;

impl Foo {
    pub fn bar() -> Foo {
        Foo
    }
}

fn main() {}
