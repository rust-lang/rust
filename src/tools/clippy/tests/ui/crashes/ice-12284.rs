#![allow(incomplete_features)]
#![feature(unnamed_fields)]

#[repr(C)]
struct Foo {
    _: struct {
    },
}

fn main() {}
