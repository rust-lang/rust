#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct Source<'a> {
    data: &'static mut (),
    marker: PhantomData<&'a ()>,
}

impl<'a> Reborrow for Source<'a> {}

struct Target<'a> {
    data: &'a (),
    //~^ ERROR implementing `CoerceShared` requires corresponding fields to match
}

impl<'a> CoerceShared<Target<'a>> for Source<'a> {}

fn main() {}
