use crate::marker::PhantomData;
use crate::slice;

#[repr(C)]
#[derive(Clone, Copy)]
struct WSABUF {
    len: u32,
    buf: *mut u8,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub(super) struct IoSlice<'a> {
    vec: WSABUF,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub(super) fn new(buf: &'a [u8]) -> IoSlice<'a> {
        assert!(buf.len() <= u32::MAX as usize);
        IoSlice {
            vec: WSABUF { len: buf.len() as u32, buf: buf.as_ptr() as *mut u8 },
            _p: PhantomData,
        }
    }

    #[inline]
    pub(super) fn advance(&mut self, n: usize) {
        if (self.vec.len as usize) < n {
            panic!("advancing IoSlice beyond its length");
        }

        // SAFETY:
        //  * `n <= len` as asserted above.
        //  * The allocation pointed to by `buf` is valid up to `buf + len`.
        unsafe {
            self.vec.len -= n as u32;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub(super) const fn as_slice(&self) -> &'a [u8] {
        // SAFETY:
        //  * `buf` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts(self.vec.buf, self.vec.len as usize) }
    }
}

#[repr(transparent)]
pub(super) struct IoSliceMut<'a> {
    vec: WSABUF,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub(super) fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        assert!(buf.len() <= u32::MAX as usize);
        IoSliceMut { vec: WSABUF { len: buf.len() as u32, buf: buf.as_mut_ptr() }, _p: PhantomData }
    }

    #[inline]
    pub(super) fn advance(&mut self, n: usize) {
        if (self.vec.len as usize) < n {
            panic!("advancing IoSliceMut beyond its length");
        }

        // SAFETY:
        //  * `n <= len` as asserted above.
        //  * The allocation pointed to by `buf` is valid up to `buf + len`.
        unsafe {
            self.vec.len -= n as u32;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[u8] {
        // SAFETY:
        //  * `buf` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts(self.vec.buf, self.vec.len as usize) }
    }

    #[inline]
    pub(super) const fn into_slice(self) -> &'a mut [u8] {
        // SAFETY:
        //  * `buf` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts_mut(self.vec.buf, self.vec.len as usize) }
    }

    #[inline]
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY:
        //  * `buf` and `len` come from a prior decomposition of a valid slice.
        unsafe { slice::from_raw_parts_mut(self.vec.buf, self.vec.len as usize) }
    }
}
