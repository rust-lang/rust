//! Buffer management for same-process client<->server communication.

use std::io::{self, Write};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice;

#[repr(C)]
struct Slice<'a, T> {
    data: &'a [T; 0],
    len: usize,
}

unsafe impl<'a, T: Sync> Sync for Slice<'a, T> {}
unsafe impl<'a, T: Sync> Send for Slice<'a, T> {}

impl<T> Copy for Slice<'a, T> {}
impl<T> Clone for Slice<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> From<&'a [T]> for Slice<'a, T> {
    fn from(xs: &'a [T]) -> Self {
        Slice {
            data: unsafe { &*(xs.as_ptr() as *const [T; 0]) },
            len: xs.len(),
        }
    }
}

impl<T> Deref for Slice<'a, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
}

#[repr(C)]
pub struct Buffer<T: Copy> {
    data: *mut T,
    len: usize,
    capacity: usize,
    extend_from_slice: extern "C" fn(Buffer<T>, Slice<'_, T>) -> Buffer<T>,
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

    pub(super) fn extend_from_slice(&mut self, xs: &[T]) {
        // Fast path to avoid going through an FFI call.
        if let Some(final_len) = self.len.checked_add(xs.len()) {
            if final_len <= self.capacity {
                let dst = unsafe { slice::from_raw_parts_mut(self.data, self.capacity) };
                dst[self.len..][..xs.len()].copy_from_slice(xs);
                self.len = final_len;
                return;
            }
        }
        let b = self.take();
        *self = (b.extend_from_slice)(b, Slice::from(xs));
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
                let Buffer {
                    data,
                    len,
                    capacity,
                    ..
                } = b;
                mem::forget(b);
                Vec::from_raw_parts(data, len, capacity)
            }
        }

        extern "C" fn extend_from_slice<T: Copy>(b: Buffer<T>, xs: Slice<'_, T>) -> Buffer<T> {
            let mut v = to_vec(b);
            v.extend_from_slice(&xs);
            Buffer::from(v)
        }

        extern "C" fn drop<T: Copy>(b: Buffer<T>) {
            mem::drop(to_vec(b));
        }

        Buffer {
            data,
            len,
            capacity,
            extend_from_slice,
            drop,
        }
    }
}
