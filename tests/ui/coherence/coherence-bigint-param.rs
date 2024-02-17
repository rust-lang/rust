//@ aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote1;

pub struct BigInt;

impl<T> Remote1<BigInt> for T { }
//~^ ERROR E0210

fn main() { }
