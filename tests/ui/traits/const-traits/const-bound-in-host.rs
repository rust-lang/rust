//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl, effects)]
//~^ WARN the feature `effects` is incomplete

#[const_trait] trait Foo {
    fn foo();
}

fn foo<T: const Foo>() {
    const { T::foo() }
}

fn main() {}
