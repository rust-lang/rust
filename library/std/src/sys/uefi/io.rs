use crate::ffi::c_void;
use crate::marker::PhantomData;
use crate::slice;

#[derive(Copy, Clone)]
pub struct IoSlice<'a> {
    vec: UefiBuf,
    _p: PhantomData<&'a [u8]>,
}

impl<'a> IoSlice<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> IoSlice<'a> {
        let len = buf.len().try_into().unwrap();
        IoSlice {
            vec: UefiBuf { len, buf: buf.as_ptr() as *mut u8 as *mut c_void },
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        let n_u32 = n.try_into().unwrap();
        if self.vec.len < n_u32 {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.len -= n_u32;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.buf as *mut u8, self.vec.len as usize) }
    }
}

pub struct IoSliceMut<'a> {
    vec: UefiBuf,
    _p: PhantomData<&'a mut [u8]>,
}

impl<'a> IoSliceMut<'a> {
    #[inline]
    pub fn new(buf: &'a mut [u8]) -> IoSliceMut<'a> {
        let len = buf.len().try_into().unwrap();
        IoSliceMut { vec: UefiBuf { len, buf: buf.as_mut_ptr().cast() }, _p: PhantomData }
    }

    #[inline]
    pub fn advance(&mut self, n: usize) {
        let n_u32 = n.try_into().unwrap();
        if self.vec.len < n_u32 {
            panic!("advancing IoSlice beyond its length");
        }

        unsafe {
            self.vec.len -= n_u32;
            self.vec.buf = self.vec.buf.add(n);
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.vec.buf as *mut u8, self.vec.len as usize) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.vec.buf as *mut u8, self.vec.len as usize) }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
struct UefiBuf {
    pub len: u32,
    pub buf: *mut c_void,
}
