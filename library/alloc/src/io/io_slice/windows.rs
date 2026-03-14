use core::marker::PhantomData;
use core::slice;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct WSABUF {
    pub len: u32,
    pub buf: *mut u8,
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct IoSlice<'a> {
    vec: WSABUF,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        assert!(buf.len() <= u32::MAX as usize);
        IoSlice {
            vec: WSABUF { len: buf.len() as u32, buf: buf.as_ptr() as *mut u8 },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if (self.vec.len as usize) < n {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.len -= n as u32;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub const fn as_slice(&self) -> &'a [u8] {
        unsafe { slice::from_raw_parts(self.vec.buf, self.vec.len as usize) }
    }
}

#[repr(transparent)]
pub struct IoSliceMut<'a> {
    vec: WSABUF,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        assert!(buf.len() <= u32::MAX as usize);
        IoSliceMut { vec: WSABUF { len: buf.len() as u32, buf: buf.as_mut_ptr() }, _p: PhantomData }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        if (self.vec.len as usize) < n {
            panic!("advancing IoSliceMut beyond its length");
        }

        unsafe {
            self.vec.len -= n as u32;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.buf, self.vec.len as usize) }
    }

    #[inline]
    pub const fn into_slice(self) -> &'a mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.buf, self.vec.len as usize) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.buf, self.vec.len as usize) }
    }
}
