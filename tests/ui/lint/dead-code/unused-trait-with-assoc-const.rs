#![deny(dead_code)]

trait Tr { //~ ERROR trait `Tr` is never used
    const I: Self;
}

struct Foo; //~ ERROR struct `Foo` is never constructed

impl Tr for Foo {
    const I: Self = Foo;
}

fn main() {}
