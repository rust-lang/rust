//@ compile-flags: -Zvalidate-mir
//@ check-pass

// Check that we don't cause cycle errors when validating pre-`RevealOpaques` MIR
// that assigns opaques through normalized projections.

#![feature(impl_trait_in_assoc_type)]

struct Bar;

trait Trait {
    type Assoc;
    fn foo() -> Foo;
}

impl Trait for Bar {
    type Assoc = impl std::fmt::Debug;
    fn foo() -> Foo
    where
        Self::Assoc:,
    {
        let x: <Bar as Trait>::Assoc = ();
        Foo { field: () }
    }
}

struct Foo {
    field: <Bar as Trait>::Assoc,
}

fn main() {}
