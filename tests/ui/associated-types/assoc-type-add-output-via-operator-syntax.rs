//! Regression test for https://github.com/rust-lang/rust/issues/20803

//@ run-pass
use std::ops::Add;

fn foo<T>(x: T) -> <i32 as Add<T>>::Output where i32: Add<T> {
    42i32 + x
}

fn main() {
    println!("{}", foo(0i32));
}
