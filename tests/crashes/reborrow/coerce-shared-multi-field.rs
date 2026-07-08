//@ known-bug: unknown
#![feature(reborrow)]
#![allow(dead_code)]

use std::marker::{CoerceShared, PhantomData, Reborrow};
use std::ptr::NonNull;

struct MatMut<'a, T> {
    ptr: NonNull<T>,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> Reborrow for MatMut<'a, T> {}

struct MatRef<'a, T> {
    ptr: NonNull<T>,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    marker: PhantomData<&'a T>,
}

impl<'a, T> Clone for MatRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for MatRef<'a, T> {}

impl<'a, T> CoerceShared<MatRef<'a, T>> for MatMut<'a, T> {}

fn dims<T>(mat: MatRef<'_, T>) -> (usize, usize, usize, usize) {
    let _ = mat.ptr;
    (mat.rows, mat.cols, mat.row_stride, mat.col_stride)
}

fn main() {
    let mut value = 0;
    let mat = MatMut {
        ptr: NonNull::from(&mut value),
        rows: 2,
        cols: 3,
        row_stride: 4,
        col_stride: 5,
        marker: PhantomData,
    };

    assert_eq!(dims(mat), (2, 3, 4, 5));
    // Reusing the same source proves repeated shared reborrows keep source-only data protected
    // without consuming the reborrowable value.
    assert_eq!(dims(mat), (2, 3, 4, 5));
}
