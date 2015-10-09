// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use c::prelude::*;
use alloc::boxed::Box;
use collections::borrow::{Borrow, Cow, ToOwned};
use collections::{String, Vec};
use core::mem;
use core::str;
use core::slice;
use core::ops::Deref;
use core::cmp::Ordering;

#[derive(PartialEq, PartialOrd, Eq, Ord, Hash, Clone)]
pub struct CString {
    inner: Box<[u8]>,
}

#[derive(Hash)]
pub struct CStr {
    inner: [c_char]
}

#[derive(Clone, PartialEq, Debug)]
pub struct NulError(usize, Vec<u8>);

impl CString {
    pub fn new<T: Into<Vec<u8>>>(t: T) -> Result<CString, NulError> {
        Self::_new(t.into())
    }

    fn _new(bytes: Vec<u8>) -> Result<CString, NulError> {
        match bytes.iter().position(|x| *x == 0) {
            Some(i) => Err(NulError(i, bytes)),
            None => Ok(unsafe { CString::from_vec_unchecked(bytes) }),
        }
    }

    pub unsafe fn from_vec_unchecked(mut v: Vec<u8>) -> CString {
        v.push(0);
        CString { inner: v.into_boxed_slice() }
    }

    pub unsafe fn from_ptr(ptr: *const c_char) -> CString {
        CString::from_raw(ptr as *mut _)
    }

    pub unsafe fn from_raw(ptr: *mut c_char) -> CString {
        let len = strlen(ptr) + 1; // Including the NUL byte
        let slice = slice::from_raw_parts(ptr, len as usize);
        CString { inner: mem::transmute(slice) }
    }

    pub fn into_ptr(self) -> *const c_char {
        self.into_raw() as *const _
    }

    pub fn into_raw(self) -> *mut c_char {
        Box::into_raw(self.inner) as *mut c_char
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.inner[..self.inner.len() - 1]
    }

    pub fn as_bytes_with_nul(&self) -> &[u8] {
        &self.inner
    }
}

impl Deref for CString {
    type Target = CStr;

    fn deref(&self) -> &CStr {
        unsafe { mem::transmute(self.as_bytes_with_nul()) }
    }
}

impl Borrow<CStr> for CString {
    fn borrow(&self) -> &CStr { self }
}

impl NulError {
    pub fn nul_position(&self) -> usize { self.0 }

    pub fn into_vec(self) -> Vec<u8> { self.1 }
}

impl CStr {
    pub unsafe fn from_ptr<'a>(ptr: *const c_char) -> &'a CStr {
        let len = strlen(ptr);
        mem::transmute(slice::from_raw_parts(ptr, len as usize + 1))
    }

    pub fn as_ptr(&self) -> *const c_char {
        self.inner.as_ptr()
    }

    pub fn to_bytes(&self) -> &[u8] {
        let bytes = self.to_bytes_with_nul();
        &bytes[..bytes.len() - 1]
    }

    pub fn to_bytes_with_nul(&self) -> &[u8] {
        unsafe { mem::transmute(&self.inner) }
    }

    pub fn to_str(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(self.to_bytes())
    }

    pub fn to_string_lossy(&self) -> Cow<str> {
        String::from_utf8_lossy(self.to_bytes())
    }
}

impl PartialEq for CStr {
    fn eq(&self, other: &CStr) -> bool {
        self.to_bytes().eq(other.to_bytes())
    }
}
impl Eq for CStr {}
impl PartialOrd for CStr {
    fn partial_cmp(&self, other: &CStr) -> Option<Ordering> {
        self.to_bytes().partial_cmp(&other.to_bytes())
    }
}
impl Ord for CStr {
    fn cmp(&self, other: &CStr) -> Ordering {
        self.to_bytes().cmp(&other.to_bytes())
    }
}

impl ToOwned for CStr {
    type Owned = CString;

    fn to_owned(&self) -> CString {
        unsafe { CString::from_vec_unchecked(self.to_bytes().to_vec()) }
    }
}
