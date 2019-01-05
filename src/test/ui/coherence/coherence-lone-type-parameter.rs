// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_lib as lib;
use lib::Remote;

impl<T> Remote for T { }
//[old]~^ ERROR type parameter `T` must be used as the type parameter for some local type
//[re]~^^ ERROR E0210


fn main() { }
