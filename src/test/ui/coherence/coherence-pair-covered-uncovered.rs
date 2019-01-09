// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_lib as lib;
use lib::{Remote, Pair};

struct Local<T>(T);

impl<T,U> Remote for Pair<T,Local<U>> { }
//[old]~^ ERROR type parameter `T` must be used as the type parameter for some local type
//[re]~^^ ERROR E0117

fn main() { }
