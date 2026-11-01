//@ run-pass
#![allow(unused_variables)]
// Test that overloaded index expressions with DST result types
// work and don't ICE.

use std::ops::Index;
use std::fmt::Debug;

struct S;

impl Index<usize> for S {
    type Output = str;

    fn index<'a>(&'a self, _: usize) -> &'a str {
        "hello"
    }
}

struct T;

impl Index<usize> for T {
    type Output = dyn Debug + 'static;

    fn index<'a>(&'a self, idx: usize) -> &'a (dyn Debug + 'static) {
        static X: usize = 42;
        &X as &(dyn Debug + 'static)
    }
}

fn main() {
    assert_eq!(&S[0], "hello");
    let _ = &T[0];
    // let x = &x as &Debug;
}
