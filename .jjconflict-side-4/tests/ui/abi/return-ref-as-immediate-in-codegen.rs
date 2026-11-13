//! Regression test for <https://github.com/rust-lang/rust/issues/38727>
//@ build-pass
#![allow(dead_code)]
#[repr(u64)]
enum A {
    A = 0u64,
    B = !0u64,
}

fn cmp() -> A {
    A::B
}

fn main() {}
