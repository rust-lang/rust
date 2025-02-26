//@ revisions: simple negative_coherence

#![feature(negative_impls)]
#![cfg_attr(negative_coherence, feature(with_negative_coherence))]

trait MyTrait {}

impl<T: Copy> MyTrait for T { }

impl MyTrait for String { }
//~^ ERROR conflicting implementations of trait `MyTrait` for type `String`

fn main() {}
