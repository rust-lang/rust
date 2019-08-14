// run-pass
#![allow(dead_code)]
// Test that inherent impls can be defined for extern types.

#![feature(extern_types)]

extern {
    type A;
}

impl A {
    fn foo(&self) { }
}

fn use_foo(x: &A) {
    x.foo();
}

fn main() { }
