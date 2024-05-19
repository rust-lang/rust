//@ check-pass
#![allow(dead_code)]
// regression test for #19097

struct Foo<T>(T);

impl<'a, T> Foo<&'a T> {
    fn foo(&self) {}
}
impl<'a, T> Foo<&'a mut T> {
    fn foo(&self) {}
}

fn main() {}
