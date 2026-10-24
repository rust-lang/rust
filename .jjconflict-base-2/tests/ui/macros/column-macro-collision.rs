//! regression test for <https://github.com/rust-lang/rust/issues/43057>
//! user-defined `column!` macro must not shadow
//! the built-in `column!()` used internally by `panic!()`.
//@ check-pass
#![allow(unused)]

macro_rules! column {
    ($i:ident) => {
        $i
    };
}

fn foo() -> ! {
    panic!();
}

fn main() {}
