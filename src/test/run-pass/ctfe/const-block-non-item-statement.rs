#![feature(const_let)]

enum Foo {
    Bar = { let x = 1; 3 }
}

pub fn main() {}
