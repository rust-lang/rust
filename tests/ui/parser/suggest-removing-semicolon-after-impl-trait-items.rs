//@ run-rustfix
#![allow(dead_code)]

trait Foo {
    fn bar() {}; //~ ERROR non-item in item list
}

fn main() {}
