//@ normalize-stderr: "\n\n\z" -> "\n"

#![feature(reborrow)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct GenericMut<'a, T, U> {
    value: &'a mut T,
    marker: PhantomData<U>,
}

impl<'a, T, U> Reborrow for GenericMut<'a, T, U> {}

#[derive(Clone, Copy)]
struct GenericRef<'a, T, U> {
    value: &'a U,
    //~^ ERROR
    marker: PhantomData<T>,
}

impl<'a, T, U> CoerceShared<GenericRef<'a, T, U>> for GenericMut<'a, T, U> {}

fn main() {}
