use crate::marker::PhantomData;
use crate::slice;
use crate::sys::c;

#[repr(transparent)]
pub struct IoSlice<'a> {
    vec: c::WSABUF,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        assert!(buf.len() <= c::ULONG::max_value() as usize);
        IoSlice {
            vec: c::WSABUF {
                len: buf.len() as c::ULONG,
                buf: buf.as_ptr() as *mut u8 as *mut c::CHAR,
            },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.vec.buf as *mut u8, self.vec.len as usize)
        }
    }
}

pub struct IoSliceMut<'a> {
    vec: c::WSABUF,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        assert!(buf.len() <= c::ULONG::max_value() as usize);
        IoSliceMut {
            vec: c::WSABUF {
                len: buf.len() as c::ULONG,
                buf: buf.as_mut_ptr() as *mut c::CHAR,
            },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(self.vec.buf as *mut u8, self.vec.len as usize)
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            slice::from_raw_parts_mut(self.vec.buf as *mut u8, self.vec.len as usize)
        }
    }
}
