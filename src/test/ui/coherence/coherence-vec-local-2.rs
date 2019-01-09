// Test that a local, generic type appearing within a
// *non-fundamental* remote type like `Vec` is not considered local.

// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

extern crate coherence_lib as lib;
use lib::Remote;

struct Local<T>(T);

impl<T> Remote for Vec<Local<T>> { }
//[old]~^ ERROR E0210
//[re]~^^ ERROR E0117

fn main() { }
