// Regression test for #137190.
// Variant of vtable-unsatisfied-supertrait.rs with generic traits.
// Verify that we don't ICE when building vtable entries
// for a generic trait whose supertrait is not implemented.

//@ compile-flags: --crate-type lib

trait Supertrait<T> {
    fn method(&self) {}
}

trait Trait<P>: Supertrait<()> {}

impl<P> Trait<P> for () {}
//~^ ERROR the trait bound `(): Supertrait<()>` is not satisfied

const fn upcast<P>(x: &dyn Trait<P>) -> &dyn Supertrait<()> {
    x
}

const fn foo() -> &'static dyn Supertrait<()> {
    upcast::<()>(&())
}

const _: &'static dyn Supertrait<()> = foo();
