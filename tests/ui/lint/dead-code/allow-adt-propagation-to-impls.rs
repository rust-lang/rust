//@ check-pass

#![deny(dead_code)]

pub trait Tr {
    fn foo(&self);
}

#[allow(dead_code)]
struct Foo;

impl Tr for Foo {
    fn foo(&self) {
        bar();
    }
}

fn bar() {}

fn main() {}
