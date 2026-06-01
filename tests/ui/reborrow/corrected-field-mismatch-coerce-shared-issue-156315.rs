//@ normalize-stderr: "\n\n\z" -> "\n"

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
