//@ check-pass
// This test should compile without an ICE.
#![expect(incomplete_features)]
#![feature(gca_min_const_items)]

use std::gca;

const CONST: usize = gca!(1);

fn uses_const() {
    CONST;
}

fn main() {}
