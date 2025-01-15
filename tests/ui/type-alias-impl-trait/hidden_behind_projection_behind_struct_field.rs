//! This test shows that a field type that is a projection that resolves to an opaque,
//! is not a defining use. While we could substitute the struct generics, that would
//! mean we would have to walk all substitutions of an `Foo`, which can quickly
//! degenerate into looking at an exponential number of types depending on the complexity
//! of a program.

#![feature(impl_trait_in_assoc_type)]

struct Bar;

trait Trait: Sized {
    type Assoc;
    fn foo() -> Foo<Self>;
}

impl Trait for Bar {
    type Assoc = impl std::fmt::Debug;
    //~^ ERROR: unconstrained opaque type
    fn foo() -> Foo<Bar> {
        Foo { field: () }
        //~^ ERROR: mismatched types
    }
}

struct Foo<T: Trait> {
    field: <T as Trait>::Assoc,
}

fn main() {}
