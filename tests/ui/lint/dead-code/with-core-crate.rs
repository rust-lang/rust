#![deny(dead_code)]
#![allow(unreachable_code)]

#[macro_use]
extern crate core;

fn foo() { //~ ERROR function `foo` is never used

    // none of these should have any dead_code exposed to the user
    panic!();

    panic!("foo");

    panic!("bar {}", "baz")
}


fn main() {}
