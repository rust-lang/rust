// Tests that we consider `i16: Remote` to be ambiguous, even
// though the upstream crate doesn't implement it for now.

// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]


extern crate coherence_lib;

use coherence_lib::Remote;

struct A<X>(X);
impl<T> A<T> where T: Remote { fn dummy(&self) { } }
//[old]~^ ERROR E0592
//[re]~^^ ERROR E0592
impl A<i16> { fn dummy(&self) { } }

fn main() {}
