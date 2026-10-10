//@ compile-flags: -Znext-solver=globally
// Regression test for issue https://github.com/rust-lang/rust/issues/151330

trait Supertrait<T> {}

trait Trait<P>: Supertrait {}
//~^ ERROR missing generics for trait `Supertrait`
//~| ERROR missing generics for trait `Supertrait`
//~| ERROR missing generics for trait `Supertrait`

impl<P> Trait<P> for () {}

const fn upcast<P>(x: &dyn Trait<P>) -> &dyn Trait<P> {
    x
}

const fn foo() -> &'static dyn Supertrait<()> {
    upcast::<()>(&())
}

const _: &'static dyn Supertrait<()> = foo();

fn main() {}
