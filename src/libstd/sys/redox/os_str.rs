// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The underlying OsString/OsStr implementation on Unix systems: just
/// a `Vec<u8>`/`[u8]`.

use borrow::Cow;
use fmt::{self, Debug};
use str;
use mem;
use sys_common::{AsInner, IntoInner};

#[derive(Clone, Hash)]
pub struct Buf {
    pub inner: Vec<u8>
}

pub struct Slice {
    pub inner: [u8]
}

impl Debug for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.to_string_lossy().fmt(formatter)
    }
}

impl Debug for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.as_slice().fmt(formatter)
    }
}

impl IntoInner<Vec<u8>> for Buf {
    fn into_inner(self) -> Vec<u8> {
        self.inner
    }
}

impl AsInner<[u8]> for Buf {
    fn as_inner(&self) -> &[u8] {
        &self.inner
    }
}


impl Buf {
    pub fn from_string(s: String) -> Buf {
        Buf { inner: s.into_bytes() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf {
            inner: Vec::with_capacity(capacity)
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    pub fn as_slice(&self) -> &Slice {
        unsafe { mem::transmute(&*self.inner) }
    }

    pub fn into_string(self) -> Result<String, Buf> {
        String::from_utf8(self.inner).map_err(|p| Buf { inner: p.into_bytes() } )
    }

    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.extend_from_slice(&s.inner)
    }
}

impl Slice {
    fn from_u8_slice(s: &[u8]) -> &Slice {
        unsafe { mem::transmute(s) }
    }

    pub fn from_str(s: &str) -> &Slice {
        Slice::from_u8_slice(s.as_bytes())
    }

    pub fn to_str(&self) -> Option<&str> {
        str::from_utf8(&self.inner).ok()
    }

    pub fn to_string_lossy(&self) -> Cow<str> {
        String::from_utf8_lossy(&self.inner)
    }

    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_vec() }
    }
}
