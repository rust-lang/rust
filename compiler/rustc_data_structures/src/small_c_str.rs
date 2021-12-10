use std::ffi;
use std::ops::Deref;

use smallvec::SmallVec;

#[cfg(test)]
mod tests;

const SIZE: usize = 36;

/// Like SmallVec but for C strings.
#[derive(Clone)]
pub struct SmallCStr {
    data: SmallVec<[u8; SIZE]>,
}

impl SmallCStr {
    #[inline]
    pub fn new(s: &str) -> SmallCStr {
        let len = s.len();
        let len1 = len + 1;
        let data = if len < SIZE {
            let mut buf = [0; SIZE];
            buf[..len].copy_from_slice(s.as_bytes());
            SmallVec::from_buf_and_len(buf, len1)
        } else {
            let mut data = Vec::with_capacity(len1);
            data.extend_from_slice(s.as_bytes());
            data.push(0);
            SmallVec::from_vec(data)
        };
        if let Err(e) = ffi::CStr::from_bytes_with_nul(&data) {
            panic!("The string \"{}\" cannot be converted into a CStr: {}", s, e);
        }
        SmallCStr { data }
    }

    #[inline]
    pub fn new_with_nul(s: &str) -> SmallCStr {
        let b = s.as_bytes();
        if let Err(e) = ffi::CStr::from_bytes_with_nul(b) {
            panic!("The string \"{}\" cannot be converted into a CStr: {}", s, e);
        }
        SmallCStr { data: SmallVec::from_slice(s.as_bytes()) }
    }

    #[inline]
    pub fn as_c_str(&self) -> &ffi::CStr {
        unsafe { ffi::CStr::from_bytes_with_nul_unchecked(&self.data) }
    }

    #[inline]
    pub fn len_with_nul(&self) -> usize {
        self.data.len()
    }

    pub fn spilled(&self) -> bool {
        self.data.spilled()
    }
}

impl Deref for SmallCStr {
    type Target = ffi::CStr;

    fn deref(&self) -> &ffi::CStr {
        self.as_c_str()
    }
}
