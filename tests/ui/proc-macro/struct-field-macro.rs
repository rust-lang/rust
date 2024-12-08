//@ run-pass

#![allow(dead_code)]
//@ proc-macro: derive-nothing.rs

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
