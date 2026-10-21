#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct Source<'a> {
    a: &'a mut u8,
    b: u8,
}

#[derive(Copy, Clone)]
struct Target<'a> {
    a: &'a u8,
    b: u8,
}

impl Reborrow for Source<'_> {}

impl<'a: 'b, 'b> CoerceShared<Target<'b>> for Source<'a> {}
//~^ ERROR implementing `CoerceShared` currently requires source and target to have at most one non-ZST reborrow data field

fn main() {}
