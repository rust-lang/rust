//! Regression test for https://github.com/rust-lang/rust/issues/23406

//@ build-pass
#![allow(dead_code)]
trait Inner {
    type T;
}

impl<'a> Inner for &'a i32 {
    type T = i32;
}

fn f<'a>(x: &'a i32) -> <&'a i32 as Inner>::T {
    *x
}

fn main() {}
