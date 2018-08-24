// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote;

struct Foo;

impl<T> Remote for lib::Pair<T,Foo> { }
//~^ ERROR type parameter `T` must be used as the type parameter for some local type

fn main() { }
