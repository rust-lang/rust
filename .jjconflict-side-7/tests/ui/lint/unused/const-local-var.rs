// regression test for https://github.com/rust-lang/rust/issues/69016
//@ check-pass

#![warn(unused)]
#![deny(warnings)]

fn _unused1(x: i32) -> i32 {
    const F: i32 = 2;
    let g = 1;
    x * F + g
}

pub struct Foo {}

impl Foo {
    fn _unused2(x: i32) -> i32 {
        const F: i32 = 2;
        let g = 1;
        x * F + g
    }
}

fn main() {}
