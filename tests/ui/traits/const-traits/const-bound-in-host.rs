//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]

#[const_trait] trait Foo {
    fn foo();
}

fn foo<T: const Foo>() {
    const { T::foo() }
}

fn main() {}
