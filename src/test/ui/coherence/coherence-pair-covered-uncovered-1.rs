// Test that the same coverage rules apply even if the local type appears in the
// list of type parameters, not the self type.

// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]


extern crate coherence_lib as lib;
use lib::{Remote1, Pair};

pub struct Local<T>(T);

impl<T, U> Remote1<Pair<T, Local<U>>> for i32 { }
//[old]~^ ERROR type parameter `T` must be used as the type parameter for some local type
//[re]~^^ ERROR E0117

fn main() { }
