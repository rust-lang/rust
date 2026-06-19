#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct Source<'a> {
    a: &'a mut u8,
    b: u8,
}

struct Target<'a> {
    a: &'a u8,
    b: u8,
    //~^ ERROR
}

impl Reborrow for Source<'_> {}

impl<'a> CoerceShared<Target<'a>> for Source<'a> {}

fn main() {}
