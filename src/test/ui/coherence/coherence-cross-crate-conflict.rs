// The error here is strictly due to orphan rules; the impl here
// generalizes the one upstream

// aux-build:trait_impl_conflict.rs
extern crate trait_impl_conflict;
use trait_impl_conflict::Foo;

impl<A> Foo for A {
    //~^ ERROR type parameter `A` must be used as the type parameter for some local type
    //~| ERROR conflicting implementations of trait `trait_impl_conflict::Foo` for type `isize`
}

fn main() {
}
