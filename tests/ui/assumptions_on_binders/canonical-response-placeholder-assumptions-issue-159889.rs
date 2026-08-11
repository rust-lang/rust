//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![feature(type_alias_impl_trait)]

// Regression test for #159889. Normalizing `FooAssoc<'a>` returns an opaque type constraint
// containing a query-created placeholder. Applying that response recreates its universe in the
// caller, where it must have an assumptions entry before eager placeholder handling visits it.
//
// Adding `impl Trait for ()` to this same shape still ICEs on nightly (same
// `assumptions` unwrap). A separately rewritten check-pass does not hit this path, so it is not
// a useful regression guard.

trait Trait {
    type Assoc<'a>;
}

type Foo = impl for<'a> Trait<Assoc<'a> = FooAssoc<'a>>;
type FooAssoc<'a> = impl ?Sized;

#[define_opaque(Foo)]
fn foo() -> Foo {}
//~^ ERROR the trait bound `(): Trait` is not satisfied

fn main() {}
