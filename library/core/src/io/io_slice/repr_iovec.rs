use crate::ffi::c_void;
use crate::marker::PhantomData;
use crate::slice;

#[derive(Copy, Clone)]
#[repr(C)]
struct iovec {
    iov_base: *mut c_void,
    iov_len: usize,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(super) struct IoSlice<'a> {
    vec: iovec,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub(super) fn new(buf: &'a [u8]) -> IoSlice<'a> {
        IoSlice {
            vec: iovec { iov_base: buf.as_ptr() as *mut u8 as *mut c_void, iov_len: buf.len() },
            _p: PhantomData,
        }
    }

    #[inline]
    pub(super) fn advance(&mut self, n: usize) {
        if self.vec.iov_len < n {
            panic!("advancing IoSlice beyond its length");
        }

        // SAFETY:
        //  * `n <= iov_len` as asserted above.
        //  * The allocation pointed to by `iov_base` is valid up to `iov_base + iov_len`.
        unsafe {
            self.vec.iov_len -= n;
            self.vec.iov_base = self.vec.iov_base.add(n);
        }
    }

    #[inline]
    pub(super) const fn as_slice(&self) -> &'a [u8] {
        // SAFETY:
        //  * `iov_base` and `iov_len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts(self.vec.iov_base as *mut u8, self.vec.iov_len) }
    }
}

#[repr(transparent)]
pub(super) struct IoSliceMut<'a> {
    vec: iovec,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub(super) fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        IoSliceMut {
            vec: iovec { iov_base: buf.as_mut_ptr() as *mut c_void, iov_len: buf.len() },
            _p: PhantomData,
        }
    }

    #[inline]
    pub(super) fn advance(&mut self, n: usize) {
        if self.vec.iov_len < n {
            panic!("advancing IoSliceMut beyond its length");
        }

        // SAFETY:
        //  * `n <= iov_len` as asserted above.
        //  * The allocation pointed to by `iov_base` is valid up to `iov_base + iov_len`.
        unsafe {
            self.vec.iov_len -= n;
            self.vec.iov_base = self.vec.iov_base.add(n);
        }
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[u8] {
        // SAFETY:
        //  * `iov_base` and `iov_len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts(self.vec.iov_base as *mut u8, self.vec.iov_len) }
    }

    #[inline]
    pub(super) const fn into_slice(self) -> &'a mut [u8] {
        // SAFETY:
        //  * `iov_base` and `iov_len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts_mut(self.vec.iov_base as *mut u8, self.vec.iov_len) }
    }

    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY:
        //  * `iov_base` and `iov_len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts_mut(self.vec.iov_base as *mut u8, self.vec.iov_len) }
    }
}
