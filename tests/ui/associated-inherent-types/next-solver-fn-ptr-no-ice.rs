//! regression test for <https://github.com/rust-lang/rust/issues/142717>
//!
//! note: the next trait solver no longer ICEs here. The old solver still does, see
//! `tests/crashes/142717.rs`.
//@ compile-flags: -Znext-solver=globally

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl<'a> Foo<fn(&'a ())> {
    type Assoc = &mut Peekable<I>;
    //~^ ERROR `&` without an explicit lifetime name cannot be used here
    //~| ERROR cannot find type `Peekable` in this scope
    //~| ERROR cannot find type `I` in this scope
}

fn bar(_: for<'a> fn(Foo<fn(Foo<fn(&'a ())>::Assoc)>::Assoc)) {}

fn main() {}
