//! Regression test for https://github.com/rust-lang/rust/issues/22992

//@ run-pass
struct A(B);
struct B;

use std::ops::Deref;

impl Deref for A {
    type Target = B;
    fn deref(&self) -> &B { &self.0 }
}

impl B {
    fn foo(&self) {}
}

fn main() {
    A(B).foo();
}
