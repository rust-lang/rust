use crate::marker::PhantomData;
use crate::slice;

use libc::{__wasi_ciovec_t, __wasi_iovec_t, c_void};

#[repr(transparent)]
pub struct IoSlice<'a> {
    vec: __wasi_ciovec_t,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        IoSlice {
            vec: __wasi_ciovec_t {
                buf: buf.as_ptr() as *const c_void,
                buf_len: buf.len(),
            },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.vec.buf as *const u8, self.vec.buf_len)
        }
    }
}

pub struct IoSliceMut<'a> {
    vec: __wasi_iovec_t,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        IoSliceMut {
            vec: __wasi_iovec_t {
                buf: buf.as_mut_ptr() as *mut c_void,
                buf_len: buf.len()
            },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.vec.buf as *const u8, self.vec.buf_len)
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(self.vec.buf as *mut u8, self.vec.buf_len)
        }
    }
}
