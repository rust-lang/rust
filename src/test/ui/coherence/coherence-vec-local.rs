// Test that a local type (with no type parameters) appearing within a
// *non-fundamental* remote type like `Vec` is not considered local.

// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_lib as lib;
use lib::Remote;

struct Local;

impl Remote for Vec<Local> { }
//[old]~^ ERROR E0117
//[re]~^^ ERROR E0117

fn main() { }
