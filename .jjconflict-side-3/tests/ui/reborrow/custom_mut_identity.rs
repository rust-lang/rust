//@ run-pass

#![feature(reborrow)]
use std::marker::Reborrow;

#[allow(unused)]
struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}

fn method(a: CustomMut<()>) -> CustomMut<()> {
    a
}

fn main() {
    let a = CustomMut(&mut ());
    let _ = method(a);
}
