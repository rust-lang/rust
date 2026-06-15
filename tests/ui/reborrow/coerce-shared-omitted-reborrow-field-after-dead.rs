//@ check-pass

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
}

impl<'a, T> Clone for OmitRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for OmitRef<'a, T> {}

impl<'a, T> CoerceShared<OmitRef<'a, T>> for OmitMut<'a, T> {}

fn read(value: OmitRef<'_, i32>) {
    assert_eq!(*value.value, 1);
}

fn main() {
    let mut value = 1;
    let mut extra_value = 2;

    {
        let extra = ExtraMut { value: &mut extra_value };
        let wrapped = OmitMut { value: &mut value, extra };

        read(wrapped);
    }

    extra_value = 3;
    assert_eq!(extra_value, 3);
}
