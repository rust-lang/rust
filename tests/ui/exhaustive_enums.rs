// run-rustfix

#![deny(clippy::exhaustive_enums)]
#![allow(unused)]

fn main() {
    // nop
}

enum Exhaustive {
    Foo,
    Bar,
    Baz,
    Quux(String),
}

#[non_exhaustive]
enum NonExhaustive {
    Foo,
    Bar,
    Baz,
    Quux(String),
}
