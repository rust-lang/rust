// Test that we check usage sites of lazy type aliases, aka free alias types, for well-formedness.
// We check trailing where-clauses separately in `trailing-where-clause.rs`.

#![feature(lazy_type_alias)]

type B<T: Copy> = T;

fn b<X>(_: B<X>) {}
//~^ ERROR the trait bound `X: Copy` is not satisfied
//~| ERROR the trait bound `X: Copy` is not satisfied

type A<'a: 'static> = &'a ();

fn a<'r>(_: A<'r>) {} //~ ERROR lifetime bound not satisfied

fn main() {}
