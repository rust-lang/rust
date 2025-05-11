//@ check-pass

#![deny(clippy::if_same_then_else)]

// Test for https://github.com/rust-lang/rust-clippy/issues/2426

fn main() {}

pub fn foo(a: i32, b: i32) -> Option<&'static str> {
    if a == b {
        None
    } else if a > b {
        Some("a pfeil b")
    } else {
        None
    }
}
