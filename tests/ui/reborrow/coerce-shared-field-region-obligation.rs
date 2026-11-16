#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct FieldMut<'a, T> {
    value: &'a mut T,
}

impl<'a, T> Reborrow for FieldMut<'a, T> {}

#[derive(Clone, Copy)]
struct FieldRef<'a, T> {
    value: &'a T,
}

impl<'a, T> CoerceShared<FieldRef<'a, T>> for FieldMut<'a, T> {}

struct Source<'a> {
    field: FieldMut<'a, &'a ()>,
}

impl Reborrow for Source<'_> {}

#[derive(Clone, Copy)]
struct Target<'a> {
    field: FieldRef<'a, &'static ()>,
}

impl<'a: 'b, 'b> CoerceShared<Target<'b>> for Source<'a> {}
//~^ ERROR mismatched types
//~| ERROR cannot infer an appropriate lifetime for lifetime parameter `'a` due to conflicting requirements [E0803]

fn main() {}
