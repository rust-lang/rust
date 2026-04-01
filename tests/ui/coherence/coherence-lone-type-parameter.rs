//@ aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote;

impl<T> Remote for T { }
//~^ ERROR E0210


fn main() { }
