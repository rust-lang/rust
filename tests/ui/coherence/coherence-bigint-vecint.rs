//@ run-pass
//@ aux-build:coherence_lib.rs

//@ pretty-expanded FIXME #23616

extern crate coherence_lib as lib;
use lib::Remote1;

pub struct BigInt;

impl Remote1<BigInt> for Vec<isize> { }

fn main() { }
