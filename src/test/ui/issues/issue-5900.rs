// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

pub mod foo {
    use super::Bar;

    pub struct FooStruct { bar : Bar }
}

pub enum Bar {
    Bar0 = 0 as isize
}

pub fn main() {}
