//! regression test for <https://github.com/rust-lang/rust/issues/142717>
//!
//! note: the next trait solver no longer ICEs here. The old solver still does, see
//! `tests/crashes/142717.rs`.
//@ compile-flags: -Znext-solver=globally

#![feature(inherent_associated_types)]

struct Type<T>(T);

impl<'a> Type<fn(&'a ())> {
    type Assoc = ();
}

fn bar(_: for<'a> fn(Type<fn(Type<fn(&'a ())>::Assoc)>::Assoc)) {}
//~^ ERROR type mismatch resolving `Type<fn(())>::Assoc
//~| ERROR type mismatch resolving `Type<fn(())>::Assoc
//~| ERROR type mismatch resolving `Type<fn(())>::Assoc
//~| ERROR type mismatch resolving `Type<fn(&())>::Assoc

fn main() {}
