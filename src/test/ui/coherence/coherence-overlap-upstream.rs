// Tests that we consider `i16: Remote` to be ambiguous, even
// though the upstream crate doesn't implement it for now.

// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]


extern crate coherence_lib;

use coherence_lib::Remote;

trait Foo {}
impl<T> Foo for T where T: Remote {}
impl Foo for i16 {}
//[old]~^ ERROR E0119
//[re]~^^ ERROR E0119

fn main() {}
