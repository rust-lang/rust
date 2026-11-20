// The error here is strictly due to orphan rules; the impl here
// generalizes the one upstream

//@ aux-build:trait_impl_conflict.rs

extern crate trait_impl_conflict;
use trait_impl_conflict::Foo;

impl<A> Foo for A { //~ ERROR E0210
}

fn main() {
}
