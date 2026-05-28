//! regression test for <https://github.com/rust-lang/rust/issues/24389>
//@ check-pass
#![allow(dead_code)]

struct Foo;

impl Foo {
    fn new() -> Self {
        Foo
    }
    fn bar() {
        Self::new();
    }
}

fn main() {}
