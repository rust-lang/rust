//@ check-pass

#![warn(clippy::use_self)]

#[macro_use]
#[path = "auxiliary/use_self_macro.rs"]
mod use_self_macro;

struct Foo {
    a: u32,
}

use_self! {
    impl Foo {
        fn func(&self) {
            [fields(
                a
            )]
        }
    }
}

fn main() {}
