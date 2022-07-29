// run-pass
// aux-build:add-impl.rs

#![forbid(unused_qualifications)]

#[macro_use]
extern crate add_impl;

#[derive(AddImpl)]
struct B;

fn main() {
    B.foo();
    foo();
    bar::foo();
}
