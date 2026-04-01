// Fixes #119830

#![feature(min_specialization)]
#![feature(const_trait_impl)]

trait Specialize {}

trait Foo {}

impl<T> const Foo for T {}
//~^ error: const `impl` for trait `Foo` which is not `const`

impl<T> const Foo for T where T: const Specialize {}
//~^ error: const `impl` for trait `Foo` which is not `const`
//~| error: `const` can only be applied to `const` traits
//~| error: specialization impl does not specialize any associated items
//~| error: cannot specialize on trait `Specialize`

fn main() {}
