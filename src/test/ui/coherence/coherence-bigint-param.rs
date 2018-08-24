// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote1;

pub struct BigInt;

impl<T> Remote1<BigInt> for T { }
//~^ ERROR type parameter `T` must be used as the type parameter for some local type

fn main() { }
