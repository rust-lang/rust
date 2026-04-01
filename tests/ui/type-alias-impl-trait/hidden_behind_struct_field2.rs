//! This test shows that we do not treat opaque types
//! as defined by a method if the opaque type is
//! only indirectly mentioned in a struct field.

#![feature(impl_trait_in_assoc_type)]

struct Bar;

trait Trait: Sized {
    type Assoc;
    fn foo() -> Foo;
}

impl Trait for Bar {
    type Assoc = impl std::fmt::Debug;
    //~^ ERROR: unconstrained opaque type
    fn foo() -> Foo {
        Foo { field: () }
        //~^ ERROR: mismatched types
    }
}

struct Foo {
    field: <Bar as Trait>::Assoc,
}

fn main() {}
