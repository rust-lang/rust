//@compile-flags: -Zvalidate-mir -Zinline-mir=yes --crate-type=lib

#![feature(async_drop)]
#![allow(incomplete_features)]

use std::{
    future::{Future, async_drop_in_place},
    pin::pin,
    task::Context,
};

fn wrong() -> impl Sized {
    //~^ ERROR: the size for values of type `str` cannot be known at compilation time
    *"abc" // Doesn't implement Sized
}
fn weird(context: &mut Context<'_>) {
    let mut e = wrong();
    let h = unsafe { async_drop_in_place(&raw mut e) };
    let i = pin!(h);
    i.poll(context);
}
