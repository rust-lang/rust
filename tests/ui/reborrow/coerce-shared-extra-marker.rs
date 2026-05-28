//@ run-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, PhantomData, Reborrow};

struct MarkerExtraMut<'a, T> {
    value: &'a mut T,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> Reborrow for MarkerExtraMut<'a, T> {}

struct MarkerExtraRef<'a, T> {
    value: &'a T,
}

impl<'a, T> Clone for MarkerExtraRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for MarkerExtraRef<'a, T> {}

impl<'a, T> CoerceShared<MarkerExtraRef<'a, T>> for MarkerExtraMut<'a, T> {}

fn get<'a>(value: MarkerExtraRef<'a, i32>) -> &'a i32 {
    value.value
}

fn main() {
    let mut value = 1;
    let wrapped = MarkerExtraMut { value: &mut value, marker: PhantomData };
    assert_eq!(*get(wrapped), 1);
}
