#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

struct MissingSourceMut<'a, T> {
    value: &'a mut T,
}

impl<'a, T> Reborrow for MissingSourceMut<'a, T> {}

#[derive(Clone, Copy)]
struct MissingSourceRef<'a, T> {
    value: &'a T,
    len: usize,
    //~^ ERROR
}

impl<'a, T> CoerceShared<MissingSourceRef<'a, T>> for MissingSourceMut<'a, T> {}

fn main() {}
