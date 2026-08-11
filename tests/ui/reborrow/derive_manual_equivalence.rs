//@ run-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct DerivedRef<'a, T>(&'a T);
impl<'a, T> Clone for DerivedRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for DerivedRef<'a, T> {}

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(DerivedRef<'a, T>)]
struct DerivedMut<'a, T>(&'a mut T);

struct ManualRef<'a, T>(&'a T);
impl<'a, T> Clone for ManualRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for ManualRef<'a, T> {}

struct ManualMut<'a, T>(&'a mut T);

impl<'a, T> Reborrow for ManualMut<'a, T> {}
impl<'a, T> CoerceShared<ManualRef<'a, T>> for ManualMut<'a, T> {}

fn take_derived(_: DerivedRef<'_, ()>) {}
fn take_manual(_: ManualRef<'_, ()>) {}

fn main() {
    let derived = DerivedMut(&mut ());
    take_derived(derived);

    let manual = ManualMut(&mut ());
    take_manual(manual);
}
