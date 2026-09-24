//@ check-pass
// This test should compile without an ICE.
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

use std::gca;

const CONST: usize = gca!(1);

fn uses_const() {
    CONST;
}

fn main() {}
