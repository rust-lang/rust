// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
pub struct Foo {
    x: isize,
    y: isize
}

impl Foo {
    pub extern fn foo_new() -> Foo {
        Foo { x: 21, y: 33 }
    }
}

fn main() {}
