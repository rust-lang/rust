// run-pass
#![allow(dead_code)]
// aux-build:coherence_lib.rs
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]

// pretty-expanded FIXME #23616

extern crate coherence_lib as lib;
use lib::Remote;

struct Foo<T>(T);

impl<T> Remote for Foo<T> { }

fn main() { }
