//@ check-pass
#![allow(dead_code)]

struct Foo;

impl Foo {
    fn new() -> Self { Foo }
    fn bar() { Self::new(); }
}

fn main() {}
