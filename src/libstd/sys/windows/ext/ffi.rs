// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows-specific extensions to the primitives in the `std::ffi` module.

#![stable(feature = "rust1", since = "1.0.0")]

use ffi::{OsString, OsStr};
use sys::os_str::Buf;
use sys_common::wtf8::Wtf8Buf;
use sys_common::{FromInner, AsInner};

#[stable(feature = "rust1", since = "1.0.0")]
pub use sys_common::wtf8::EncodeWide;

/// Windows-specific extensions to `OsString`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt {
    /// Creates an `OsString` from a potentially ill-formed UTF-16 slice of
    /// 16-bit code units.
    ///
    /// This is lossless: calling [`encode_wide`] on the resulting string
    /// will always return the original code units.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    /// ```
    ///
    /// [`encode_wide`]: ./trait.OsStrExt.html#tymethod.encode_wide
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_wide(wide: &[u16]) -> Self;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    fn from_wide(wide: &[u16]) -> OsString {
        FromInner::from_inner(Buf { inner: Wtf8Buf::from_wide(wide) })
    }
}

/// Windows-specific extensions to `OsStr`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt {
    /// Re-encodes an `OsStr` as a wide character sequence, i.e. potentially
    /// ill-formed UTF-16.
    ///
    /// This is lossless: calling [`OsString::from_wide`] and then
    /// `encode_wide` on the result will yield the original code units.
    /// Note that the encoding does not add a final null terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    ///
    /// let result: Vec<u16> = string.encode_wide().collect();
    /// assert_eq!(&source[..], &result[..]);
    /// ```
    ///
    /// [`OsString::from_wide`]: ./trait.OsStringExt.html#tymethod.from_wide
    #[stable(feature = "rust1", since = "1.0.0")]
    fn encode_wide(&self) -> EncodeWide;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    fn encode_wide(&self) -> EncodeWide {
        self.as_inner().inner.encode_wide()
    }
}
