//@ known-bug: unknown
#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, PhantomData, Reborrow};
use std::ptr::NonNull;

struct ImbrisMut<'a, T> {
    ptr: NonNull<T>,
    metadata: usize,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> Reborrow for ImbrisMut<'a, T> {}

struct ImbrisRef<'a, T> {
    ptr: NonNull<T>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> Clone for ImbrisRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for ImbrisRef<'a, T> {}

impl<'a, T> CoerceShared<ImbrisRef<'a, T>> for ImbrisMut<'a, T> {}

fn ptr(value: ImbrisRef<'_, i32>) -> NonNull<i32> {
    value.ptr
}

fn main() {
    let mut value = 1;
    let raw = NonNull::from(&mut value);
    let wrapped = ImbrisMut { ptr: raw, metadata: 32, marker: PhantomData };

    assert_eq!(ptr(wrapped), raw);
}
