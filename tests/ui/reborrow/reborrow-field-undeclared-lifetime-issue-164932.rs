//! Regression test for <https://github.com/rust-lang/rust/issues/164932>.
//!
//! A `Reborrow` impl on a type whose field uses an undeclared lifetime
//! used to cause an ICE in borrowck instead of an error.

#![feature(reborrow)]

use std::marker::Reborrow;

struct CustomMut<'b, T>(&'a mut T);
//~^ ERROR use of undeclared lifetime name `'a`

impl<'a, T> Reborrow for CustomMut<'a, T> {}

fn method(a: CustomMut<()>) -> CustomMut<()> {
    a
}

fn main() {}
