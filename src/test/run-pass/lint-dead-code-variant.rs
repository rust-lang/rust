#![deny(dead_code)]

enum Foo {
    A,
    B,
}

pub fn main() {
    match Foo::A {
        Foo::A | Foo::B => Foo::B
    };
}
