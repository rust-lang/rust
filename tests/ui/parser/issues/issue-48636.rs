//@ run-rustfix

#![allow(dead_code)]

struct S {
    x: u8
    /// The ID of the parent core
    y: u8,
}
//~^^^ ERROR found a documentation comment that doesn't document anything

fn main() {}
