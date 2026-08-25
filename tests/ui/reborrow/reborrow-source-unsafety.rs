// Regression test for rust-lang/rust#158033.

#![feature(reborrow)]
#![allow(dead_code)]
#![deny(unsafe_code)]

use std::marker::Reborrow;

struct Thing<'a> {
    field: &'a mut usize,
}

impl<'a> Reborrow for Thing<'a> {}

fn takes(_: Thing<'_>) {}

fn main() {
    let mut x = 0;
    let thing = Thing { field: &mut x };
    let y = 123usize;

    takes({
        let p: *const usize = &y;
        std::hint::black_box(std::ptr::read(p));
        //~^ ERROR call to unsafe function `std::ptr::read` is unsafe
        thing
    });
}
