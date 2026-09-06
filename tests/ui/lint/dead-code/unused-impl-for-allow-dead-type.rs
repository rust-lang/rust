//@ check-pass

#![deny(dead_code)]

#[allow(dead_code)]
struct Foo;

impl Foo {
    fn foo(&self) {}
}

pub trait Tr {
    fn foo(&self);
}

impl Tr for Foo {
    fn foo(&self) {
        bar()
    }
}

fn bar() {}

fn main() {}
