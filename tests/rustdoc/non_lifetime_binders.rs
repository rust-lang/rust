#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

pub trait Trait {}

pub struct Wrapper<T, const N: usize>([T; N]);

// @has non_lifetime_binders/fn.foo.html '//pre' "fn foo()where for<'a, T, const N: usize> &'a Wrapper<T, N>: Trait"
pub fn foo() where for<'a, T, const N: usize> &'a Wrapper<T, N>: Trait {}
