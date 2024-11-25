//@ run-pass
//@ aux-build:coherence_lib.rs


extern crate coherence_lib as lib;
use lib::Remote1;

pub struct BigInt;

impl Remote1<BigInt> for isize { }

fn main() { }
