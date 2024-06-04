//@ check-fail
// Fixes #119830

#![feature(effects)]
#![feature(min_specialization)]
#![feature(const_trait_impl)]

trait Specialize {}

trait Foo {}

impl<T> const Foo for T {}
//~^ error: const `impl` for trait `Foo` which is not marked with `#[const_trait]`
//~| error: the const parameter `host` is not constrained by the impl trait, self type, or predicates [E0207]

impl<T> const Foo for T where T: const Specialize {}
//~^ error: const `impl` for trait `Foo` which is not marked with `#[const_trait]`
//~| error: `const` can only be applied to `#[const_trait]` traits
//~| error: the const parameter `host` is not constrained by the impl trait, self type, or predicates [E0207]
//~| error: specialization impl does not specialize any associated items
//~| error: could not resolve generic parameters on overridden impl

fn main() {
}
