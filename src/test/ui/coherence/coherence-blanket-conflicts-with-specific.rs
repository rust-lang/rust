// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

use std::fmt::Debug;
use std::default::Default;

// Test that a blank impl for all T conflicts with an impl for some
// specific T.

trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for T {
    fn get(&self) -> usize { 0 }
}

struct MyType {
    dummy: usize
}

impl MyTrait for MyType {
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119
    fn get(&self) -> usize { self.dummy }
}

fn main() { }
