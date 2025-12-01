//@ run-pass
//! regression test for <https://github.com/rust-lang/rust/issues/3290>

#![allow(dead_code)]

pub fn main() {
    let mut x: Box<_> = Box::new(3);
    x = x;
    assert_eq!(*x, 3);
}
