//@ check-pass
#![feature(reborrow)]

use std::marker::Reborrow;

#[allow(unused)]
struct Thing<'a>(&'a ());

impl<'a> Reborrow for Thing<'a> {}

fn main() {
    let x = Thing(&());
    let _y: Thing<'_> = x;
}
