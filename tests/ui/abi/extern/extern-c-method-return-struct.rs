//! Regression test for <https://github.com/rust-lang/rust/issues/26997>.
//@ build-pass

#![allow(dead_code)]

#[repr(C)]
pub struct Foo {
    x: isize,
    y: isize
}

impl Foo {
    pub extern "C" fn foo_new() -> Foo {
        Foo { x: 21, y: 33 }
    }
}

fn main() {}
