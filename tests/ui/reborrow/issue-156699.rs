#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct Source<'a>(&'a mut ());

impl<'a> Reborrow for Source<'a> {}

impl<'a, T> CoerceShared<<T as Iterator>::Item> for Source<'a> {}
//~^ ERROR `T` is not an iterator
//~| ERROR the type parameter `T` is not constrained

trait CustomMarker<'a> {}

impl<'a, T> CoerceShared<<T as Iterator>::Item> for dyn CustomMarker<'a> {}
//~^ ERROR `T` is not an iterator
//~| ERROR the type parameter `T` is not constrained

fn main() {}
