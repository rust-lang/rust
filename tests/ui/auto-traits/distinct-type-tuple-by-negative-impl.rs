//! Regression test for <https://github.com/rust-lang/rust/issues/29516>
//@ check-pass
#![feature(auto_traits)]
#![feature(negative_impls)]

auto trait NotSame {}

impl<A> !NotSame for (A, A) {}

trait OneOfEach {}

impl<A> OneOfEach for (A,) {}

impl<A, B> OneOfEach for (A, B)
where
    (B,): OneOfEach,
    (A, B): NotSame,
{
}

fn main() {}
