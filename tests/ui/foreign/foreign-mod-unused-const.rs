//@ run-pass
#![allow(dead_code)]

mod foo {
    extern "C" {
        pub static errno: u32;
    }
}

pub fn main() {}
