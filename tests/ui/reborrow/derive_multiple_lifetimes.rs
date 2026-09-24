//! This test mirrors manual and derived impls with multiple lifetimes. The derive
//! layer preserves both lifetimes; the current underlying trait validation rejects
//! both forms with the same experimental one-lifetime limitation.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};


struct ManualPair<'a, 'b, T> {
    left: &'a mut T,
    right: &'b mut T,
}

impl<'a, 'b, T> Reborrow for ManualPair<'a, 'b, T> {}

#[derive(Reborrow)]
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
impl<'a, 'b, T> CoerceShared<ManualShared<'a, 'b, T>> for ManualMut<'a, 'b, T> {}
//~^ ERROR implementing `CoerceShared` currently requires source and target to have at most one non-ZST reborrow data field

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
//~^ ERROR implementing `CoerceShared` currently requires source and target to have at most one non-ZST reborrow data field
#[coerce_shared(DerivedShared<'a, 'b, T>)]
struct DerivedMut<'a, 'b, T> {
    left: &'a mut T,
    right: &'b mut T,
}

fn main() {}
