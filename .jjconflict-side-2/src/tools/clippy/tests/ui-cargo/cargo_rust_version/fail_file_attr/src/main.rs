// FIXME: this should produce a warning, because the attribute says 1.58 and the cargo.toml file
// says 1.13

#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.58.0"]
#![deny(clippy::use_self)]

pub struct Foo;

impl Foo {
    pub fn bar() -> Foo {
        Foo
    }
}

fn main() {}
