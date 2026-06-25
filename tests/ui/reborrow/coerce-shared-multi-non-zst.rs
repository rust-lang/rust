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

impl<'a> CoerceShared<Target<'a>> for Source<'a> {}
//~^ ERROR

fn main() {}
