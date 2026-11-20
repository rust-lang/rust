#![feature(reborrow)]

use std::marker::Reborrow;

#[derive(Reborrow)]
enum Enum<'a, T> {
    A(&'a mut T),
}

fn method(_: Enum<'_, ()>) {}

fn main() {
    let a = Enum::A(&mut ());
    method(a);
    method(a);
}
