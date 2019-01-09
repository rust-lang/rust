// The error here is strictly due to orphan rules; the impl here
// generalizes the one upstream

// aux-build:trait_impl_conflict.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate trait_impl_conflict;
use trait_impl_conflict::Foo;

impl<A> Foo for A {
    //[old]~^ ERROR type parameter `A` must be used as the type parameter for some local type
    //[old]~| ERROR conflicting implementations of trait `trait_impl_conflict::Foo` for type `isize`
    //[re]~^^^ ERROR E0119
    //[re]~| ERROR E0210
}

fn main() {
}
