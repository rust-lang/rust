//! Regression test for https://github.com/rust-lang/rust/issues/6341.
//@ check-pass

#[derive(PartialEq)]
struct A {
    x: usize,
}

impl Drop for A {
    fn drop(&mut self) {}
}

pub fn main() {}
