// build-pass
#![allow(dead_code)]
#![allow(improper_ctypes)]
// pretty-expanded FIXME #23616
#![allow(non_snake_case)]

pub mod Bar {
    pub struct Foo {
        v: isize,
    }

    extern "C" {
        pub fn foo(v: *const Foo) -> Foo;
    }
}

pub fn main() {}
