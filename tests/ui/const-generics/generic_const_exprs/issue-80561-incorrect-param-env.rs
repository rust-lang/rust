//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// This tests that the correct `param_env` is used so that
// attempting to normalize `Self::N` does not cause an ICE.

pub struct Foo<const N: usize>;

impl<const N: usize> Foo<N> {
    pub fn foo() {}
}

pub trait Bar {
    const N: usize;
    fn bar()
    where
        [(); Self::N]: ,
    {
        Foo::<{ Self::N }>::foo();
    }
}

fn main() {}
