//@ run-pass

#![feature(reborrow)]
use std::marker::Reborrow;

#[allow(unused)]
struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}

fn method(_: CustomMut<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    let _ = method(a);
    let _ = method(a);
}
