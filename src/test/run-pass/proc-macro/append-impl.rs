// aux-build:append-impl.rs

#![allow(warnings)]

#[macro_use]
extern crate append_impl;

trait Append {
    fn foo(&self);
}

#[derive(PartialEq,
         Append,
         Eq)]
struct A {
    inner: u32,
}

fn main() {
    A { inner: 3 }.foo();
}
