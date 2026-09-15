#![feature(transmutability)]
trait OpaqueTrait {}

impl<T: std::mem::TransmuteFrom<(), ()>> OpaqueTrait for T {}
//~^ ERROR: type provided when a constant was expected

impl<T> OpaqueTrait for &T where T: OpaqueTrait {}
//~^ ERROR: conflicting implementations of trait `OpaqueTrait` for type `&_`

fn main() {}
