//! Test that CoerceShared cannot produce a field from thin air.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct CustomMut<'a, T> {
    value: &'a mut T,
}

impl<'a, T> Reborrow for CustomMut<'a, T> {}

#[derive(Clone, Copy)]
struct CustomRef<'a, T> {
    value: &'a T,
}

// No error expected here: value: &'a mut T -> value: &'a T.
impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

struct RenamedMut<'a, T> {
    source: &'a mut T,
}

impl<'a, T> Reborrow for RenamedMut<'a, T> {}

#[derive(Clone, Copy)]
struct RenamedRef<'a, T> {
    target: &'a T,
}

// Should error: source: &'a mut T -> target: &'a T attempts to drop 'source' and produce
// 'target' from thin air.
impl<'a, T> CoerceShared<RenamedRef<'a, T>> for RenamedMut<'a, T> {}

struct BadMut<'a, T> {
    value: &'a mut T,
}

impl<'a, T> Reborrow for BadMut<'a, T> {}

#[derive(Clone, Copy)]
struct BadRef<'a, T> {
    value: &'a u32,
    _marker: std::marker::PhantomData<T>,
}

// Should error: value: &'a mut T -> &'a u32 attempts a reference transmute, and also
// '_marker' field is created from thin air.
impl<'a, T> CoerceShared<BadRef<'a, T>> for BadMut<'a, T> {}
//~^ ERROR

fn good(_value: CustomRef<'_, u32>) {}

fn main() {
    let mut value = 1;
    good(CustomMut { value: &mut value });
}
