//! regression test for <https://github.com/rust-lang/rust/issues/141014>
//@ run-pass
#![expect(incomplete_features)]
#![feature(gca_min_const_items, gca_macroless_args)]
#![allow(dead_code)]

use std::gca;

trait Abc {}

trait A {
    #[rustc_always_gca]
    const VALUE: usize;
}

impl<T: Abc> A for T {
    const VALUE: usize = gca!(0);
}

trait S<const K: usize> {}

trait Handler<T: Abc>
where
    (): S<{ <T as A>::VALUE }>,
{
}

fn main() {}
