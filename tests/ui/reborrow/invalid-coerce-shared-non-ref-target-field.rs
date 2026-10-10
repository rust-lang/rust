//! Regression test for an ICE in borrowck when a `CoerceShared` coercion goes through an impl
//! that maps a `&mut T` field to a field that is not a reference.
//! See <https://github.com/rust-lang/rust/issues/162142>.

//! Test that a custom mut type (a wrapper over &mut T) implementing CoerceShared can be coerced
//! into its target.

#![feature(reborrow)]
use std::marker::{CoerceShared, Reborrow};

#[allow(unused)]
struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}
impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

struct CustomRef<'a, T>([T; [16, 1usize]]);
//~^ ERROR E0308
//~| ERROR implementing `CoerceShared` requires corresponding fields to match, be reborrowable with `CoerceShared`, or coerce a mutable reference field to a shared reference field
//~| ERROR E0392

impl<'a, T> Clone for CustomRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for CustomRef<'a, T> {}

fn method(_a: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    method(a);
}
