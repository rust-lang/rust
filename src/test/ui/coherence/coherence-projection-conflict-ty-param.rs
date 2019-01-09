// Coherence error results because we do not know whether `T: Foo<P>` or not
// for the second impl.

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

use std::marker::PhantomData;

pub trait Foo<P> { fn foo() {} }

impl <P, T: Foo<P>> Foo<P> for Option<T> {}

impl<T, U> Foo<T> for Option<U> { }
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119

fn main() {}
