#![allow(dead_code)]

#[deny(overflowing_literals)]
#[repr(i64)]
enum Foo {
    X = 0x7fffffffffffffff,
    Y, //~ ERROR E0370
}

fn main() {}
