//@ known-bug: unknown
#![feature(reborrow)]

// Malformed no-ICE regression: this intentionally keeps the missing generic arguments from the
// original reproducer while the corrected test isolates the CoerceShared field error.

use std::marker::{CoerceShared, Reborrow};

struct CustomMut<'a, T>(&'a mut T);

impl<'a, T> Reborrow for CustomMut<'a, T> {}
impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

struct CustomRef<'a, T>(&'a CustomMut);
//~^ ERROR
//~| ERROR
//~| ERROR

fn method(_a: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    method(a);
}
