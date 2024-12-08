#![deny(dead_code)]
#![allow(unreachable_code)]

fn foo() { //~ ERROR function `foo` is never used

    // none of these should have any dead_code exposed to the user
    panic!();

    panic!("foo");

    panic!("bar {}", "baz")
}


fn main() {}
