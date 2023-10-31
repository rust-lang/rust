//! This test shows that we can even follow projections
//! into associated types of the same impl if they are
//! indirectly mentioned in a struct field.

#![feature(impl_trait_in_assoc_type)]
// check-pass

struct Bar;

trait Trait: Sized {
    type Assoc;
    fn foo() -> Foo;
}

impl Trait for Bar {
    type Assoc = impl std::fmt::Debug;
    fn foo() -> Foo {
        Foo { field: () }
    }
}

struct Foo {
    field: <Bar as Trait>::Assoc,
}

fn main() {}
