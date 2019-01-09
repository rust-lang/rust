// Tests that we consider `T: Sugar + Fruit` to be ambiguous, even
// though no impls are found.

// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

struct Sweet<X>(X);
pub trait Sugar {}
pub trait Fruit {}
impl<T:Sugar> Sweet<T> { fn dummy(&self) { } }
//[old]~^ ERROR E0592
//[re]~^^ ERROR E0592
impl<T:Fruit> Sweet<T> { fn dummy(&self) { } }

trait Bar<X> {}
struct A<T, X>(T, X);
impl<X, T> A<T, X> where T: Bar<X> { fn f(&self) {} }
//[old]~^ ERROR E0592
//[re]~^^ ERROR E0592
impl<X> A<i32, X> { fn f(&self) {} }

fn main() {}
