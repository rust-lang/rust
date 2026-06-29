#![feature(reborrow)]

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
}

impl<'a, T> Clone for OmitRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for OmitRef<'a, T> {}

impl<'a, T> CoerceShared<OmitRef<'a, T>> for OmitMut<'a, T> {}
//~^ ERROR

fn get<'a>(value: OmitRef<'a, i32>) -> &'a i32 {
    value.value
}

fn main() {
    let mut value = 1;
    let mut extra_value = 2;
    let extra = ExtraMut { value: &mut extra_value };

    let mut wrapped = OmitMut {
        value: &mut value,
        extra,
    };

    let shared = get(wrapped);

    *wrapped.extra.value = 3;
    //~^ ERROR cannot assign to `*wrapped.extra.value` because it is borrowed

    let _ = shared;
}
