#![feature(reborrow)]
use std::ops::{CoerceShared, Reborrow};

struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}
impl<'a, T> CoerceShared<CustomRef<'a, T>> for CustomMut<'a, T> {}

struct CustomRef<'a, T>(&'a T);

impl<'a, T> Clone for CustomRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for CustomRef<'a, T> {}

fn method(a: CustomRef<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    method(a);
}
