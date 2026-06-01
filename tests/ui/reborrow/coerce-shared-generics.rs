//@ run-pass

#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, Reborrow};

struct BufferMut<'a, T, U, const N: usize> {
    data: &'a mut [T; N],
    meta: U,
}

impl<'a, T, U: Copy, const N: usize> Reborrow for BufferMut<'a, T, U, N> {}

struct BufferRef<'a, T, U, const N: usize> {
    data: &'a [T; N],
    meta: U,
}

impl<'a, T, U: Copy, const N: usize> Clone for BufferRef<'a, T, U, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, U: Copy, const N: usize> Copy for BufferRef<'a, T, U, N> {}

impl<'a, T, U: Copy, const N: usize> CoerceShared<BufferRef<'a, T, U, N>>
    for BufferMut<'a, T, U, N>
{
}

fn inspect<const N: usize>(buffer: BufferRef<'_, u8, u16, N>) -> (usize, u16) {
    (buffer.data.len(), buffer.meta)
}

fn main() {
    let mut data = [1, 2, 3, 4];
    let buffer = BufferMut { data: &mut data, meta: 9_u16 };

    assert_eq!(inspect(buffer), (4, 9));
}
