// https://github.com/rust-lang/rust/issues/9110
//@ check-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

macro_rules! silly_macro {
    () => (
        pub mod Qux {
            pub struct Foo { x : u8 }
            pub fn bar(_foo : Foo) {}
        }
    );
}

silly_macro!();

pub fn main() {}
