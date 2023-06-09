// aux-build: alias.rs
// regression test for 108160

extern crate alias;

use alias::Wrapper;
struct Rec(Wrapper<Rec>); //~ ERROR recursive type `Rec` has infinite

fn main() {}
