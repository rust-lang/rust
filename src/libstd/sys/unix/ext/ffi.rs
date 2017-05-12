// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unix-specific extension to the primitives in the `std::ffi` module

#![stable(feature = "rust1", since = "1.0.0")]

use ffi::{OsStr, OsString};
use mem;
use sys::os_str::Buf;
use sys_common::{FromInner, IntoInner, AsInner};

/// Unix-specific extensions to `OsString`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt {
    /// Creates an [`OsString`] from a byte vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::unix::ffi::OsStringExt;
    ///
    /// let bytes = b"foo".to_vec();
    /// let os_string = OsString::from_vec(bytes);
    /// assert_eq!(os_string.to_str(), Some("foo"));
    /// ```
    ///
    /// [`OsString`]: ../../../ffi/struct.OsString.html
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_vec(vec: Vec<u8>) -> Self;

    /// Yields the underlying byte vector of this [`OsString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::unix::ffi::OsStringExt;
    ///
    /// let mut os_string = OsString::new();
    /// os_string.push("foo");
    /// let bytes = os_string.into_vec();
    /// assert_eq!(bytes, b"foo");
    /// ```
    ///
    /// [`OsString`]: ../../../ffi/struct.OsString.html
    #[stable(feature = "rust1", since = "1.0.0")]
    fn into_vec(self) -> Vec<u8>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    fn from_vec(vec: Vec<u8>) -> OsString {
        FromInner::from_inner(Buf { inner: vec })
    }
    fn into_vec(self) -> Vec<u8> {
        self.into_inner().inner
    }
}

/// Unix-specific extensions to `OsStr`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt {
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Creates an [`OsStr`] from a byte slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    /// use std::os::unix::ffi::OsStrExt;
    ///
    /// let bytes = b"foo";
    /// let os_str = OsStr::from_bytes(bytes);
    /// assert_eq!(os_str.to_str(), Some("foo"));
    /// ```
    ///
    /// [`OsStr`]: ../../../ffi/struct.OsStr.html
    fn from_bytes(slice: &[u8]) -> &Self;

    /// Gets the underlying byte view of the [`OsStr`] slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsStr;
    /// use std::os::unix::ffi::OsStrExt;
    ///
    /// let mut os_str = OsStr::new("foo");
    /// let bytes = os_str.as_bytes();
    /// assert_eq!(bytes, b"foo");
    /// ```
    ///
    /// [`OsStr`]: ../../../ffi/struct.OsStr.html
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_bytes(&self) -> &[u8];
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    fn from_bytes(slice: &[u8]) -> &OsStr {
        unsafe { mem::transmute(slice) }
    }
    fn as_bytes(&self) -> &[u8] {
        &self.as_inner().inner
    }
}
