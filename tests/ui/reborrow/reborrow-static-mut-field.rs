//! regression test: `Reborrow` on a type with a `&'static mut` field used to ice in borrowck.

#![feature(reborrow)]

use std::marker::Reborrow;

struct Thing<'a> {
    //~^ ERROR lifetime parameter `'a` is never used
    field: &'static mut u8,
    //~^ ERROR the trait bound `&'static mut u8: Copy` is not satisfied
}

impl<'a> Reborrow for Thing<'a> {}

fn takes(_: Thing<'_>) {}

fn main() {
    let mut x = 0;
    let thing = Thing { field: &mut x };
    takes({ thing });
}
