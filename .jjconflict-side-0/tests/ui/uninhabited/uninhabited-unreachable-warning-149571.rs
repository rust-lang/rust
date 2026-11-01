#![deny(unreachable_code)]
//@ run-pass

use std::convert::Infallible;

pub fn foo(f: impl FnOnce() -> Infallible) -> Infallible {
    f()
}

fn main() {}
