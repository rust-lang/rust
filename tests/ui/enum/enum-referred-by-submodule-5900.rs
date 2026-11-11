// https://github.com/rust-lang/rust/issues/5900
//@ check-pass
#![allow(dead_code)]

pub mod foo {
    use super::Bar;

    pub struct FooStruct { bar : Bar }
}

pub enum Bar {
    Bar0 = 0 as isize
}

pub fn main() {}
