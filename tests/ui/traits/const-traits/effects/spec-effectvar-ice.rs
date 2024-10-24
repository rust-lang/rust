// Fixes #119830

#![feature(effects)] //~ WARN the feature `effects` is incomplete
#![feature(min_specialization)]
#![feature(const_trait_impl)]

trait Specialize {}

trait Foo {}

impl<T> const Foo for T {}
//~^ error: const `impl` for trait `Foo` which is not marked with `#[const_trait]`

impl<T> const Foo for T where T: const Specialize {}
//~^ error: const `impl` for trait `Foo` which is not marked with `#[const_trait]`
//~| error: `const` can only be applied to `#[const_trait]` traits
//~| error: specialization impl does not specialize any associated items
//~| error: cannot specialize on trait `Specialize`

fn main() {}
