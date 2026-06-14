//! Buffer management for same-process client<->server communication.

use std::alloc::{self, Layout};
use std::io::{self, Write};
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use std::{mem, slice};

#[repr(C)]
pub struct Buffer {
    data: *mut u8,
    len: usize,
    capacity: usize,
}

unsafe impl Sync for Buffer {}
unsafe impl Send for Buffer {}

impl Default for Buffer {
    #[inline]
    fn default() -> Self {
        Self { data: null_mut(), len: 0, capacity: 0 }
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
            self.reserve(xs.len());
        }
        unsafe {
            xs.as_ptr().copy_to_nonoverlapping(self.data.add(self.len), xs.len());
            self.len += xs.len();
        }
    }

    #[inline]
    pub(super) fn extend_from_slice(&mut self, xs: &[u8]) {
        if xs.len() > (self.capacity - self.len) {
            self.reserve(xs.len());
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
            self.reserve(1);
        }
        unsafe {
            *self.data.add(self.len) = v;
            self.len += 1;
        }
    }

    #[inline]
    fn layout(&self) -> Layout {
        Layout::array::<u8>(self.capacity).unwrap()
    }

    #[inline(never)]
    fn reserve(&mut self, amount: usize) {
        debug_assert!(amount > 0);
        self.capacity += amount;
        assert!(self.capacity < isize::MAX as usize);
        let layout = self.layout();
        self.data = if self.data.is_null() {
            unsafe { alloc::alloc(layout) }
        } else {
            unsafe { alloc::realloc(self.data, layout, self.capacity) }
        };
        if self.data.is_null() {
            alloc::handle_alloc_error(layout);
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
    #[inline]
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                alloc::dealloc(self.data, self.layout());
            }
        }
    }
}
