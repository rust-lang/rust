// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_lib as lib;
use lib::Remote;

struct Foo;

impl<T> Remote for lib::Pair<T,Foo> { }
//[old]~^ ERROR type parameter `T` must be used as the type parameter for some local type
//[re]~^^ ERROR E0117

fn main() { }
