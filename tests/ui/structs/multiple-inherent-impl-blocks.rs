//! Regression test for <https://github.com/rust-lang/rust/issues/4228>.
//! This used to emit `duplicate definition of type` error.
//@ run-pass

struct Foo;

impl Foo {
    fn first() {}
}
impl Foo {
    fn second() {}
}

pub fn main() {
    Foo::first();
    Foo::second();
}
