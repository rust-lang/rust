//@ normalize-stderr: "\n\n$" -> "\n"

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

// This test mirrors manual and derived impls with multiple lifetimes. The derive
// layer preserves both lifetimes; the current underlying trait validation rejects
// both forms with the same experimental one-lifetime limitation.

struct ManualPair<'a, 'b, T> {
    left: &'a mut T,
    right: &'b mut T,
}

impl<'a, 'b, T> Reborrow for ManualPair<'a, 'b, T> {}
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target

#[derive(Reborrow)]
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target
struct DerivedPair<'a, 'b, T> {
    left: &'a mut T,
    right: &'b mut T,
}

struct ManualShared<'a, 'b, T> {
    left: &'a T,
    right: &'b T,
}

impl<'a, 'b, T> Clone for ManualShared<'a, 'b, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, 'b, T> Copy for ManualShared<'a, 'b, T> {}

struct ManualMut<'a, 'b, T> {
    left: &'a mut T,
    right: &'b mut T,
}

impl<'a, 'b, T> Reborrow for ManualMut<'a, 'b, T> {}
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target
impl<'a, 'b, T> CoerceShared<ManualShared<'a, 'b, T>> for ManualMut<'a, 'b, T> {}
//~^ ERROR implementing `CoerceShared` requires that a single lifetime parameter is passed between source and target

struct DerivedShared<'a, 'b, T> {
    left: &'a T,
    right: &'b T,
}

impl<'a, 'b, T> Clone for DerivedShared<'a, 'b, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, 'b, T> Copy for DerivedShared<'a, 'b, T> {}

#[derive(Reborrow, CoerceShared)]
//~^ ERROR implementing `Reborrow` requires that a single lifetime parameter is passed between source and target
//~| ERROR implementing `CoerceShared` requires that a single lifetime parameter is passed between source and target
#[coerce_shared(DerivedShared<'a, 'b, T>)]
struct DerivedMut<'a, 'b, T> {
    left: &'a mut T,
    right: &'b mut T,
}

fn main() {}
