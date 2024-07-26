#![feature(impl_trait_in_assoc_type)]

// We weren't checking that the trait and impl generics line up in the
// normalization-shortcut code in `OpaqueTypeCollector`.

use std::ops::Deref;

trait Foo {
    type Bar<'a>;

    type Baz<'a>;

    fn test<'a>() -> Self::Bar<'a>;
}

impl Foo for () {
    type Bar<'a> = impl Deref<Target = Self::Baz<'a>>;

    type Baz<T> = impl Sized;
    //~^ ERROR type `Baz` has 1 type parameter but its trait declaration has 0 type parameters
    //~| ERROR: unconstrained opaque type

    fn test<'a>() -> Self::Bar<'a> {
        &()
    }
}

fn main() {}
