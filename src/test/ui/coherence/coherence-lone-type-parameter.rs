// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote;

impl<T> Remote for T { }
//~^ ERROR type parameter `T` must be used as the type parameter for some local type

fn main() { }
