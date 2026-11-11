//! A buffer type used with `Write::write_vectored` for UEFI Networking APIs. Vectored writing to
//! File is not supported as of UEFI Spec 2.11.

use crate::marker::PhantomData;
use crate::slice;

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct IoSlice<'a> {
    len: u32,
    data: *const u8,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub(crate) const fn new(buf: &'a [u8]) -> IoSlice<'a> {
        assert!(buf.len() <= u32::MAX as usize, "slice too large for IoSlice");
        IoSlice { len: buf.len() as u32, data: buf.as_ptr(), _p: PhantomData }
    }

    #[inline]
    pub(crate) const fn advance(&mut self, n: usize) {
        assert!(n <= self.len as usize, "advancing IoSlice beyond its length");

        // SAFETY:
        //  * `n <= len` as asserted above.
        //  * The allocation pointed to by `data` is valid up to `data + len`.
        unsafe {
            self.len -= n as u32;
            self.data = self.data.add(n);
        }
    }

    #[inline]
    pub(crate) const fn as_slice(&self) -> &'a [u8] {
        // SAFETY:
        //  * `data` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts(self.data, self.len as usize) }
    }
}

#[repr(C)]
pub(crate) struct IoSliceMut<'a> {
    len: u32,
    data: *mut u8,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub(crate) const fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        assert!(buf.len() <= u32::MAX as usize, "slice too large for IoSliceMut");
        IoSliceMut { len: buf.len() as u32, data: buf.as_mut_ptr(), _p: PhantomData }
    }

    #[inline]
    pub(crate) const fn advance(&mut self, n: usize) {
        assert!(n <= self.len as usize, "advancing IoSliceMut beyond its length");

        // SAFETY:
        //  * `n <= len` as asserted above.
        //  * The allocation pointed to by `data` is valid up to `data + len`.
        unsafe {
            self.len -= n as u32;
            self.data = self.data.add(n);
        }
    }

    #[inline]
    pub(crate) const fn as_slice(&self) -> &[u8] {
        // SAFETY:
        //  * `data` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts(self.data, self.len as usize) }
    }

    #[inline]
    pub(crate) const fn into_slice(self) -> &'a mut [u8] {
        // SAFETY:
        //  * `data` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts_mut(self.data, self.len as usize) }
    }

    #[inline]
    pub(crate) const fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY:
        //  * `data` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts_mut(self.data, self.len as usize) }
    }
}
