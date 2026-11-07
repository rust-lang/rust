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

fn bar() {} //~ ERROR function `bar` is never used

fn main() {}
