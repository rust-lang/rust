//! This test demonstrates a bug where we accidentally
//! detected opaque types in struct fields, but only if nested
//! in projections of another opaque type.

#![feature(impl_trait_in_assoc_type)]

struct Bar;

trait Trait: Sized {
    type Assoc2;
    type Assoc;
    fn foo() -> Self::Assoc;
}

impl Trait for Bar {
    type Assoc2 = impl std::fmt::Debug;
    //~^ ERROR: unconstrained opaque type
    type Assoc = impl Iterator<Item = Foo>;
    fn foo() -> Self::Assoc {
        vec![Foo { field: () }].into_iter()
        //~^ ERROR mismatched types
    }
}

struct Foo {
    field: <Bar as Trait>::Assoc2,
}

fn main() {}
