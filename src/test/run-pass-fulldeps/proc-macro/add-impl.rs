// aux-build:add-impl.rs
// ignore-stage1

#[macro_use]
extern crate add_impl;

#[derive(AddImpl)]
struct B;

fn main() {
    B.foo();
    foo();
    bar::foo();
}
