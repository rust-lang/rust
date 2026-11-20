//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![feature(type_alias_impl_trait)]

// Regression test for #159889. Normalizing `FooAssoc<'a>` returns an opaque type constraint
// containing a query-created placeholder. Applying that response recreates its universe in the
// caller, where it must have an assumptions entry before eager placeholder handling visits it.

trait Trait {
    type Assoc<'a>: ?Sized;
}

type Foo = impl for<'a> Trait<Assoc<'a> = FooAssoc<'a>>;
type FooAssoc<'a> = impl ?Sized;

impl Trait for () {
    type Assoc<'a> = FooAssoc<'a>;
}

#[define_opaque(Foo)]
fn foo() -> Foo {}
//~^ ERROR item does not constrain `Foo::{opaque#0}`

fn main() {}
