//@ run-pass
//@ proc-macro: add-impl.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate add_impl;

#[derive(AddImpl)]
struct B;

fn main() {
    B.foo();
    foo();
    bar::foo();
}
