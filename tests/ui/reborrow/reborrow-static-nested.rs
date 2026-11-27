#![feature(reborrow)]

use std::marker::Reborrow;

struct Inner<'a, T>(&'a (), T);

impl<'a> Reborrow for Inner<'a, &'static mut u8> {}

struct Outer<'a>(Inner<'a, &'static mut u8>);

impl<'a> Reborrow for Outer<'a> {}

fn takes(_: Outer<'_>) {}

fn main() {
    let thing = Outer(Inner(&(), Box::leak(Box::new(42))));

    takes(thing);
    takes(thing); //~ ERROR cannot borrow `thing` as mutable more than once at a time
}
