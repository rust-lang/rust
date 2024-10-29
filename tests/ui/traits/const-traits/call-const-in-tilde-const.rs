//@ compile-flags: -Znext-solver
#![feature(const_trait_impl, effects)]
//~^ WARN the feature `effects` is incomplete

#[const_trait] trait Foo {
    fn foo();
}

const fn foo<T: ~const Foo>() {
    const { T::foo() }
    //~^ ERROR the trait bound `T: const Foo` is not satisfied
}

fn main() {}
