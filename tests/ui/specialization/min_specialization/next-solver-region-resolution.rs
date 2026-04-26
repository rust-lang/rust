//@ compile-flags: -Znext-solver=globally
// Regression test for https://github.com/rust-lang/rust/issues/151327

#![feature(min_specialization)]

trait Foo {
    type Item;
}

trait Baz {}

impl<'a, T> Foo for &'a T //~ ERROR not all trait items implemented, missing: `Item`
//~| ERROR the trait bound `&'a T: Foo` is not satisfied
where
    Self::Item: 'a, //~ ERROR the trait bound `&'a T: Foo` is not satisfied
{
}

impl<'a, T> Foo for &T //~ ERROR not all trait items implemented, missing: `Item`
//~| ERROR cannot normalize `<&_ as Foo>::Item: '_`
where
    Self::Item: Baz,
{
}

fn main() {}
