// An invalid `CoerceShared` field relation must report an error instead of causing an ICE.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct CustomMut<'a, T>(&'a mut T);

impl<'a, T> Reborrow for CustomMut<'a, T> {}

struct CustomRef<'a, T>(&'a CustomMut<'a, T>);

impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}
//~^ ERROR the trait bound `&'a mut T: CoerceShared<&'a CustomMut<'a, T>>` is not satisfied

fn method(_a: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    method(a);
}
