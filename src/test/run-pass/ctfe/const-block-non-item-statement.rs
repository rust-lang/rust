// run-pass
#![allow(dead_code)]

enum Foo {
    Bar = { let x = 1; 3 }
}

pub fn main() {}
