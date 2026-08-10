//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders
//@ check-pass

#![feature(type_alias_impl_trait)]

// The original test remains the regression guard for the ICE. This check-pass
// counterpart verifies that a valid implementation of `Trait` for `()` works.

trait Trait {
    type Assoc<'a>: ?Sized;
}

type FooAssoc<'a> = impl ?Sized;

#[define_opaque(FooAssoc)]
fn foo_assoc<'a>() -> &'a FooAssoc<'a> {
    &()
}

impl Trait for () {
    type Assoc<'a> = FooAssoc<'a>;
}

fn assert_trait<T: for<'a> Trait<Assoc<'a> = FooAssoc<'a>>>() {}

fn foo() -> impl for<'a> Trait<Assoc<'a> = FooAssoc<'a>> {
    ()
}

fn main() {
    assert_trait::<()>();
    foo();
}
