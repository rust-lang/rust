#![feature(reborrow)]

use std::marker::Reborrow;

struct Wrap<'a, T>(&'a (), T);

impl<'a> Reborrow for Wrap<'a, &'static mut u8> {}

fn takes(_: Wrap<'_, &'static mut u8>) {}

fn main() {
    let thing = Wrap(&(), Box::leak(Box::new(42)));

    takes(thing);
    takes(thing); //~ ERROR cannot borrow `thing` as mutable more than once at a time
}
