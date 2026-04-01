#![feature(reborrow)]
use std::ops::Reborrow;

struct CustomMut<'a, T>(&'a mut T);
impl<'a, T> Reborrow for CustomMut<'a, T> {}

fn method(a: CustomMut<'_, ()>) {}

fn main() {
    let a = CustomMut(&mut ());
    let _ = method(a);
    let _ = method(a); //~ERROR use of moved value: `a`
}
