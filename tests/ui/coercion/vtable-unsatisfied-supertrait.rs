// Regression test for #137190.
// Verify that we don't ICE when building vtable entries
// for a trait whose supertrait is not implemented.

//@ compile-flags: --crate-type lib

trait Supertrait {
    fn method(&self) {}
}

trait Trait: Supertrait {}

impl Trait for () {}
//~^ ERROR the trait bound `(): Supertrait` is not satisfied

const _: &dyn Supertrait = &() as &dyn Trait as &dyn Supertrait;
