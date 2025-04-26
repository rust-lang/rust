//@ edition:2015
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[identity_attr]
mod m {
    pub struct X;

    type A = Y; //~ ERROR cannot find type `Y` in this scope
}

struct Y;
type A = X; //~ ERROR cannot find type `X` in this scope

#[derive(Copy)] //~ ERROR `derive` may only be applied to `struct`s, `enum`s and `union`s
mod n {}

fn main() {}
