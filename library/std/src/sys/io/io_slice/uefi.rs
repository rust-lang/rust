//! A buffer type used with `Write::write_vectored` for UEFI Networking APIs. Vectored writing to
//! File is not supported as of UEFI Spec 2.11.

use crate::marker::PhantomData;
use crate::slice;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct IoSlice<'a> {
    len: u32,
    data: *const u8,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        let len = buf.len().try_into().unwrap();
        Self { len, data: buf.as_ptr(), _p: PhantomData }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.len = u32::try_from(n)
            .ok()
            .and_then(|n| self.len.checked_sub(n))
            .expect("advancing IoSlice beyond its length");
        unsafe { self.data = self.data.add(n) };
    }

    #[inline]
    pub const fn as_slice(&self) -> &'a [u8] {
        unsafe { slice::from_raw_parts(self.data, self.len as usize) }
    }
}

#[repr(C)]
pub struct IoSliceMut<'a> {
    len: u32,
    data: *mut u8,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        let len = buf.len().try_into().unwrap();
        Self { len, data: buf.as_mut_ptr(), _p: PhantomData }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.len = u32::try_from(n)
            .ok()
            .and_then(|n| self.len.checked_sub(n))
            .expect("advancing IoSlice beyond its length");
        unsafe { self.data = self.data.add(n) };
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.data, self.len as usize) }
    }

    #[inline]
    pub const fn into_slice(self) -> &'a mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.data, self.len as usize) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.data, self.len as usize) }
    }
}
