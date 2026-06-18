#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct ExtraMut<'a, T> {
    value: &'a mut T,
}

impl<'a, T> Reborrow for ExtraMut<'a, T> {}

struct OmitMut<'a, T> {
    value: &'a mut T,
    extra: ExtraMut<'a, T>,
}

impl<'a, T> Reborrow for OmitMut<'a, T> {}

struct OmitRef<'a, T> {
    value: &'a T,
    //~^ ERROR
}

impl<'a, T> Clone for OmitRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for OmitRef<'a, T> {}

impl<'a, T> CoerceShared<OmitRef<'a, T>> for OmitMut<'a, T> {}

fn get<'a>(value: OmitRef<'a, i32>) -> &'a i32 {
    value.value
}

fn main() {
    let mut value = 1;
    let mut extra = 2;
    let extra = ExtraMut { value: &mut extra };
    let wrapped = OmitMut { value: &mut value, extra };

    assert_eq!(*get(wrapped), 1);
}
