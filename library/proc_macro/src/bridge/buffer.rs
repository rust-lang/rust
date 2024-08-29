//! Buffer management for same-process client<->server communication.

use std::io::{self, Write};
use std::mem::{self, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::slice;

#[repr(C)]
pub struct Buffer {
    data: *mut u8,
    len: usize,
    capacity: usize,
    reserve: extern "C" fn(Buffer, usize) -> Buffer,
    drop: extern "C" fn(Buffer),
}

unsafe impl Sync for Buffer {}
unsafe impl Send for Buffer {}

impl Default for Buffer {
    #[inline]
    fn default() -> Self {
        Self::from(vec![])
    }
}

impl Deref for Buffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data as *const u8, self.len) }
    }
}

impl DerefMut for Buffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.data, self.len) }
    }
}

impl Buffer {
    #[inline]
    pub(super) fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.len = 0;
    }

    #[inline]
    pub(super) fn take(&mut self) -> Self {
        mem::take(self)
    }

    // We have the array method separate from extending from a slice. This is
    // because in the case of small arrays, codegen can be more efficient
    // (avoiding a memmove call). With extend_from_slice, LLVM at least
    // currently is not able to make that optimization.
    #[inline]
    pub(super) fn extend_from_array<const N: usize>(&mut self, xs: &[u8; N]) {
        if xs.len() > (self.capacity - self.len) {
            let b = self.take();
            *self = (b.reserve)(b, xs.len());
        }
        unsafe {
            xs.as_ptr().copy_to_nonoverlapping(self.data.add(self.len), xs.len());
            self.len += xs.len();
        }
    }

    #[inline]
    pub(super) fn extend_from_slice(&mut self, xs: &[u8]) {
        if xs.len() > (self.capacity - self.len) {
            let b = self.take();
            *self = (b.reserve)(b, xs.len());
        }
        unsafe {
            xs.as_ptr().copy_to_nonoverlapping(self.data.add(self.len), xs.len());
            self.len += xs.len();
        }
    }

    #[inline]
    pub(super) fn push(&mut self, v: u8) {
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

impl Write for Buffer {
    #[inline]
    fn write(&mut self, xs: &[u8]) -> io::Result<usize> {
        self.extend_from_slice(xs);
        Ok(xs.len())
    }

    #[inline]
    fn write_all(&mut self, xs: &[u8]) -> io::Result<()> {
        self.extend_from_slice(xs);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for Buffer {
    // HACK(nbdd0121): Hack to prevent LLVM < 17.0.4 from misoptimising,
    // change to `#[inline]` if fixed.
    #[inline(never)]
    fn drop(&mut self) {
        let b = self.take();
        (b.drop)(b);
    }
}

impl From<Vec<u8>> for Buffer {
    fn from(v: Vec<u8>) -> Self {
        let mut v = ManuallyDrop::new(v);
        let (data, len, capacity) = (v.as_mut_ptr(), v.len(), v.capacity());

        // This utility function is nested in here because it can *only*
        // be safely called on `Buffer`s created by *this* `proc_macro`.
        fn to_vec(b: Buffer) -> Vec<u8> {
            unsafe {
                let b = ManuallyDrop::new(b);
                Vec::from_raw_parts(b.data, b.len, b.capacity)
            }
        }

        extern "C" fn reserve(b: Buffer, additional: usize) -> Buffer {
            let mut v = to_vec(b);
            v.reserve(additional);
            Buffer::from(v)
        }

        extern "C" fn drop(b: Buffer) {
            mem::drop(to_vec(b));
        }

        Buffer { data, len, capacity, reserve, drop }
    }
}
