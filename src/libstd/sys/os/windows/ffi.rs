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
use sys::os_str;
use sys::wtf8::Wtf8Buf;
use sys::inner::{FromInner, AsInner};

pub use sys::wtf8::EncodeWide;

/// Windows-specific extensions to `OsString`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt {
    /// Creates an `OsString` from a potentially ill-formed UTF-16 slice of
    /// 16-bit code units.
    ///
    /// This is lossless: calling `.encode_wide()` on the resulting string
    /// will always return the original code units.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_wide(wide: &[u16]) -> Self;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    fn from_wide(wide: &[u16]) -> OsString {
        FromInner::from_inner(os_str::OsString { inner: Wtf8Buf::from_wide(wide) })
    }
}

/// Windows-specific extensions to `OsStr`.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt {
    /// Re-encodes an `OsStr` as a wide character sequence,
    /// i.e. potentially ill-formed UTF-16.
    ///
    /// This is lossless. Note that the encoding does not include a final
    /// null.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn encode_wide(&self) -> EncodeWide;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    fn encode_wide(&self) -> EncodeWide {
        self.as_inner().inner.encode_wide()
    }
}
