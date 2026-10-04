//! Regression test for <https://github.com/rust-lang/rust/issues/4875>.
//! Test struct patterns as fn arguments don't ICE.
//@ run-pass
#![allow(dead_code)]


pub struct Foo<T> {
    data: T,
}

fn foo<T>(Foo{..}: Foo<T>) {
}

pub fn main() {
}
