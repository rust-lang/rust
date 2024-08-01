//@ check-pass

#![feature(trivial_bounds)]
#![allow(trivial_bounds)]

// Make sure that we still use the where-clauses from the struct when checking
// if it may implement `Copy` unconditionally.
// Fix for <https://github.com/rust-lang/rust/issues/125394>.

pub trait Foo {
    type Assoc;
}

pub struct Bar;

// This needs to be public
pub struct Baz2(<Bar as Foo>::Assoc)
where
    Bar: Foo;

fn main() {}
