// Regression test for https://github.com/rust-lang/rust/issues/162679

//@ check-pass
//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items)]
#![feature(gca_min_const_items)]

use std::marker::PhantomData;

struct S<A> {
    n: usize,
    _a: PhantomData<A>,
}

impl<A> S<A> {
    const A: usize = 1;

    fn foo(self) {
        match self.n {
            Self::A => {}
            _ => {}
        }
    }
}

fn main() {}
