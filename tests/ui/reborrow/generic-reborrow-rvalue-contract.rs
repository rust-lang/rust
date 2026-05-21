//@ check-pass

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

// Regression test for rust-lang/rust#156482.
// This used to ICE in MIR building when a THIR `ExprKind::Reborrow`
// was categorized as a place but `expr_as_place` treated it as unreachable.
struct Thing<'a>(&'a ());

impl<'a> Reborrow for Thing<'a> {}

fn foo(x: Thing<'_>) {
    let _: Thing<'_> = x;
}

#[allow(unused)]
struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}
impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

#[allow(unused)]
struct CustomRef<'a, T>(&'a T);
impl<'a, T> Clone for CustomRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for CustomRef<'a, T> {}

fn main() {
    let a = CustomMut(&mut ());

    let _: CustomMut<'_, ()> = a;
    let _: CustomMut<'_, ()> = a;

    let _: CustomRef<'_, ()> = a;
    let _: CustomRef<'_, ()> = a;
}
