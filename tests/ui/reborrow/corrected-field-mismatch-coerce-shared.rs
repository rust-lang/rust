//! Regression test for an ICE in borrowck when a `CoerceShared` coercion goes through an impl
//! whose target field does not correspond to the source field.
//! See <https://github.com/rust-lang/rust/issues/156315>.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct CustomMut<'a, T>(&'a mut T);

impl<'a, T> Reborrow for CustomMut<'a, T> {}

struct CustomRef<'a, T>(&'a CustomMut<'a, T>);
//~^ ERROR

impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

fn method(_a: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    method(a);
}
