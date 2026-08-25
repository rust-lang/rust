//@ check-pass

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

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

fn takes_mut(_: CustomMut<'_, ()>) {}
fn takes_shared(_: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());

    takes_mut(a);
    takes_mut(a);

    takes_shared(a);
    takes_shared(a);
}
