//! A `Reborrow` impl containing `&'static mut` must be rejected
//! rather than reaching an ICE in borrowck.

#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

struct Thing<'a> {
    marker: PhantomData<&'a ()>,
    field: &'static mut u8,
    //~^ ERROR the trait bound `&'static mut u8: Copy` is not satisfied
}

impl<'a> Reborrow for Thing<'a> {}

fn takes(_: Thing<'_>) {}

fn main() {
    let thing = Thing {
        marker: PhantomData,
        field: Box::leak(Box::new(42)),
    };

    takes(thing);
}
