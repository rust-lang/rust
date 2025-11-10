//@ compile-flags: -Znext-solver=coherence

#![feature(const_trait_impl)]

const trait Foo {}

impl const Foo for i32 {}

impl<T> const Foo for T where T: [const] Foo {}
//~^ ERROR conflicting implementations of trait `Foo` for type `i32`

fn main() {}
