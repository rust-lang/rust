// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

use std::fmt::Debug;
use std::default::Default;

// Test that two blanket impls conflict (at least without negative
// bounds).  After all, some other crate could implement Even or Odd
// for the same type (though this crate doesn't implement them at all).

trait MyTrait {
    fn get(&self) -> usize;
}

trait Even {}

trait Odd {}

impl<T:Even> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

impl<T:Odd> MyTrait for T {
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119
    fn get(&self) -> usize { 0 }
}

fn main() { }
