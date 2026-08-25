//@ compile-flags: -Zmir-enable-passes=+DataflowConstProp
//@ edition: 2021
//@ build-pass
#![feature(async_drop)]
#![allow(incomplete_features)]

use std::mem::ManuallyDrop;
use std::{
    future::async_drop_in_place,
    pin::{pin, Pin},
};
fn main() {
    a(b)
}
fn b() {}
fn a<C>(d: C) {
    let e = pin!(ManuallyDrop::new(d));
    let f = unsafe { Pin::map_unchecked_mut(e, |g| &mut **g) };
    let h = unsafe { async_drop_in_place(f.get_unchecked_mut()) };
    h;
}
