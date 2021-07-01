//! Buffer management for same-process client<->server communication.

use std::io::{self, Write};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice;

#[repr(C)]
pub struct Buffer<T: Copy> {
    data: *mut T,
    len: usize,
    capacity: usize,
    reserve: extern "C" fn(Buffer<T>, usize) -> Buffer<T>,
    drop: extern "C" fn(Buffer<T>),
}

unsafe impl<T: Copy + Sync> Sync for Buffer<T> {}
unsafe impl<T: Copy + Send> Send for Buffer<T> {}

impl<T: Copy> Default for Buffer<T> {
    fn default() -> Self {
        Self::from(vec![])
    }
}

impl<T: Copy> Deref for Buffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data as *const T, self.len) }
    }
}

impl<T: Copy> DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data, self.len) }
    }
}

impl<T: Copy> Buffer<T> {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn clear(&mut self) {
        self.len = 0;
    }

    pub(super) fn take(&mut self) -> Self {
        mem::take(self)
    }

    // We have the array method separate from extending from a slice. This is
    // because in the case of small arrays, codegen can be more efficient
    // (avoiding a memmove call). With extend_from_slice, LLVM at least
    // currently is not able to make that optimization.
    pub(super) fn extend_from_array<const N: usize>(&mut self, xs: &[T; N]) {
        if xs.len() > (self.capacity - self.len) {
            let b = self.take();
            *self = (b.reserve)(b, xs.len());
        }
        unsafe {
            xs.as_ptr().copy_to_nonoverlapping(self.data.add(self.len), xs.len());
            self.len += xs.len();
        }
    }

    pub(super) fn extend_from_slice(&mut self, xs: &[T]) {
        if xs.len() > (self.capacity - self.len) {
            let b = self.take();
            *self = (b.reserve)(b, xs.len());
        }
        unsafe {
            xs.as_ptr().copy_to_nonoverlapping(self.data.add(self.len), xs.len());
            self.len += xs.len();
        }
    }

    pub(super) fn push(&mut self, v: T) {
        // The code here is taken from Vec::push, and we know that reserve()
        // will panic if we're exceeding isize::MAX bytes and so there's no need
        // to check for overflow.
        if self.len == self.capacity {
            let b = self.take();
            *self = (b.reserve)(b, 1);
        }
        unsafe {
            *self.data.add(self.len) = v;
            self.len += 1;
        }
    }
}

impl Write for Buffer<u8> {
    fn write(&mut self, xs: &[u8]) -> io::Result<usize> {
        self.extend_from_slice(xs);
        Ok(xs.len())
    }

    fn write_all(&mut self, xs: &[u8]) -> io::Result<()> {
        self.extend_from_slice(xs);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<T: Copy> Drop for Buffer<T> {
    fn drop(&mut self) {
        let b = self.take();
        (b.drop)(b);
    }
}

impl<T: Copy> From<Vec<T>> for Buffer<T> {
    fn from(mut v: Vec<T>) -> Self {
        let (data, len, capacity) = (v.as_mut_ptr(), v.len(), v.capacity());
        mem::forget(v);

        // This utility function is nested in here because it can *only*
        // be safely called on `Buffer`s created by *this* `proc_macro`.
        fn to_vec<T: Copy>(b: Buffer<T>) -> Vec<T> {
            unsafe {
                let Buffer { data, len, capacity, .. } = b;
                mem::forget(b);
                Vec::from_raw_parts(data, len, capacity)
            }
        }

        extern "C" fn reserve<T: Copy>(b: Buffer<T>, additional: usize) -> Buffer<T> {
            let mut v = to_vec(b);
            v.reserve(additional);
            Buffer::from(v)
        }

        extern "C" fn drop<T: Copy>(b: Buffer<T>) {
            mem::drop(to_vec(b));
        }

        Buffer { data, len, capacity, reserve, drop }
    }
}
