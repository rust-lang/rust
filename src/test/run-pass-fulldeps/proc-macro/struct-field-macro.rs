// aux-build:derive-nothing.rs
// ignore-stage1

#[macro_use]
extern crate derive_nothing;

macro_rules! int {
    () => { i32 }
}

#[derive(Nothing)]
struct S {
    x: int!(),
}

fn main() {}
