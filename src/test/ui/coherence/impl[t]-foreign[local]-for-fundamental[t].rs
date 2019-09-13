#![feature(re_rebalance_coherence)]

// compile-flags:--crate-name=test
// aux-build:coherence_lib.rs
// check-pass

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl<T> Remote1<Local> for Box<T> {
    // FIXME(#64412) -- this is expected to error
}

impl<T> Remote1<Local> for &T {
    // FIXME(#64412) -- this is expected to error
}

fn main() {}
