//@ aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote;

struct Foo;

impl<T> Remote for lib::Pair<T,Foo> { }
//~^ ERROR E0117

fn main() { }
