//! regression test for <https://github.com/rust-lang/rust/issues/19097>
//@ check-pass
#![allow(dead_code)]

struct Foo<T>(T);

impl<'a, T> Foo<&'a T> {
    fn foo(&self) {}
}
impl<'a, T> Foo<&'a mut T> {
    fn foo(&self) {}
}

fn main() {}
