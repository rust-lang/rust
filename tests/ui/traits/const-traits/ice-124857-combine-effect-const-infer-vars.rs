//@ compile-flags: -Znext-solver=coherence

#![feature(const_trait_impl)]

const trait Foo {}

const impl Foo for i32 {}

const impl<T> Foo for T where T: [const] Foo {}
//~^ ERROR conflicting implementations of trait `Foo` for type `i32`

fn main() {}
