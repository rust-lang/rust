// run-pass
// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![allow(dead_code)]
// aux-build:coherence_lib.rs

// pretty-expanded FIXME #23616

extern crate coherence_lib as lib;
use lib::Remote1;

struct Foo<T>(T);

impl<T,U> Remote1<U> for Foo<T> { }

fn main() { }
