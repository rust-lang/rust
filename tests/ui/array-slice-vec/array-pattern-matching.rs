//! Regression test for https://github.com/rust-lang/rust/issues/10396

//@ check-pass
#![allow(dead_code)]
#[derive(Debug)]
enum Foo<'s> {
    V(&'s str)
}

fn f(arr: &[&Foo]) {
    for &f in arr {
        println!("{:?}", f);
    }
}

fn main() {}
