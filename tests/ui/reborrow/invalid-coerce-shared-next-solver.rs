//@ compile-flags: -Znext-solver=globally
//! Regression test for an ICE in borrowck under the next solver when a `CoerceShared` coercion
//! goes through an impl that coherence has already rejected.
//! See <https://github.com/rust-lang/rust/issues/156311>.

#![feature(reborrow)]
use std::marker::{CoerceShared, Reborrow};

struct CustomMut<'a, T>(&'a mut T);

impl<'a, T> CoerceShared for CustomMut<'reborrow, T> {}
//~^ ERROR E0261
//~| ERROR E0107
//~| ERROR E0377

struct CustomRef<'a, T>(&'CustomMut T);
//~^ ERROR E0261

fn method(_a: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    method(a);
}
