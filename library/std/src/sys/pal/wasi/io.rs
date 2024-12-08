#![forbid(unsafe_op_in_unsafe_fn)]

use crate::marker::PhantomData;
use crate::os::fd::{AsFd, AsRawFd};
use crate::slice;

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct IoSlice<'a> {
    vec: wasi::Ciovec,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        IoSlice { vec: wasi::Ciovec { buf: buf.as_ptr(), buf_len: buf.len() }, _p: PhantomData }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if self.vec.buf_len < n {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.buf_len -= n;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub const fn as_slice(&self) -> &'a [u8] {
        unsafe { slice::from_raw_parts(self.vec.buf as *const u8, self.vec.buf_len) }
    }
}

#[repr(transparent)]
pub struct IoSliceMut<'a> {
    vec: wasi::Iovec,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        IoSliceMut {
            vec: wasi::Iovec { buf: buf.as_mut_ptr(), buf_len: buf.len() },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if self.vec.buf_len < n {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.buf_len -= n;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.buf as *const u8, self.vec.buf_len) }
    }

    #[inline]
    pub const fn into_slice(self) -> &'a mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.buf as *mut u8, self.vec.buf_len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.buf as *mut u8, self.vec.buf_len) }
    }
}

pub fn is_terminal(fd: &impl AsFd) -> bool {
    let fd = fd.as_fd();
    unsafe { libc::isatty(fd.as_raw_fd()) != 0 }
}
