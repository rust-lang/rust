//@ aux-build:coherence_lib.rs

// issue#132826

extern crate coherence_lib;

use coherence_lib::{Pair, Remote, Remote1};

trait MyTrait {
    type Item;
}

impl<M> MyTrait for Pair<M, M> {
    type Item = Pair<M, M>;
}

impl<K> Remote for <Pair<K, K> as MyTrait>::Item {}
//~^ ERROR: the type parameter `K` is not constrained by the impl trait, self type, or predicates
//~| ERROR: only traits defined in the current crate can be implemented for arbitrary types

fn main() {}
