//! regression test for <https://github.com/rust-lang/rust/issues/141014>
//@ run-pass
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]
#![allow(dead_code)]

trait Abc {}

trait A {
    #[type_const]
    const VALUE: usize;
}

impl<T: Abc> A for T {
    #[type_const]
    const VALUE: usize = 0;
}

trait S<const K: usize> {}

trait Handler<T: Abc>
where
    (): S<{ <T as A>::VALUE }>,
{
}

fn main() {}
