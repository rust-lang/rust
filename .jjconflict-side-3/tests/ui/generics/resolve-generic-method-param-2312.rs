//! Regression test for https://github.com/rust-lang/rust/issues/2312

//@ check-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

// Testing that the B's are resolved

trait clam<A> {
    fn get(self) -> A;
}

struct foo(isize);

impl foo {
    pub fn bar<B, C: clam<B>>(&self, _c: C) -> B {
        panic!();
    }
}

pub fn main() {}
