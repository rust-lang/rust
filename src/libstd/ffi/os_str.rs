// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A type that can represent all platform-native strings, but is cheaply
//! interconvertable with Rust strings.
//!
//! The need for this type arises from the fact that:
//!
//! * On Unix systems, strings are often arbitrary sequences of non-zero
//!   bytes, in many cases interpreted as UTF-8.
//!
//! * On Windows, strings are often arbitrary sequences of non-zero 16-bit
//!   values, interpreted as UTF-16 when it is valid to do so.
//!
//! * In Rust, strings are always valid UTF-8, but may contain zeros.
//!
//! The types in this module bridge this gap by simultaneously representing Rust
//! and platform-native string values, and in particular allowing a Rust string
//! to be converted into an "OS" string with no cost.
//!
//! **Note**: At the moment, these types are extremely bare-bones, usable only
//! for conversion to/from various other string types. Eventually these types
//! will offer a full-fledged string API.

#![unstable(feature = "os_str",
            reason = "recently added as part of path/io reform")]

#[cfg(stage0)]
use core::prelude::v1::*;

use borrow::{Borrow, Cow, ToOwned};
use ffi::CString;
use fmt::{self, Debug};
use mem;
use string::String;
use ops;
use cmp;
use hash::{Hash, Hasher};
use vec::Vec;

use sys::os_str::{Buf, Slice};
use sys_common::{AsInner, IntoInner, FromInner};

/// Owned, mutable OS strings.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OsString {
    inner: Buf
}

/// Slices into OS strings.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct OsStr {
    inner: Slice
}

impl OsString {
    /// Constructs a new empty `OsString`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> OsString {
        OsString { inner: Buf::from_string(String::new()) }
    }

    /// Constructs an `OsString` from a byte sequence.
    ///
    /// # Platform behavior
    ///
    /// On Unix systems, any byte sequence can be successfully
    /// converted into an `OsString`.
    ///
    /// On Windows system, only UTF-8 byte sequences will successfully
    /// convert; non UTF-8 data will produce `None`.
    #[deprecated(reason = "use OsString::from_ill_formed_utf8() instead", since = "1.2.0")]
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn from_bytes<B>(bytes: B) -> Option<OsString> where B: Into<Vec<u8>> {
        #[cfg(unix)]
        fn from_bytes_inner(vec: Vec<u8>) -> Option<OsString> {
            use os::unix::ffi::OsStringExt;
            Some(OsString::from_vec(vec))
        }

        #[cfg(windows)]
        fn from_bytes_inner(vec: Vec<u8>) -> Option<OsString> {
            String::from_utf8(vec).ok().map(OsString::from)
        }

        from_bytes_inner(bytes.into())
    }

    /// Converts to an `OsStr` slice.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_os_str(&self) -> &OsStr {
        self
    }

    /// Converts the `OsString` into a `String` if it contains valid Unicode data.
    ///
    /// On failure, ownership of the original `OsString` is returned.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_string(self) -> Result<String, OsString> {
        self.inner.into_string().map_err(|buf| OsString { inner: buf} )
    }

    /// Extends the string with the given `&OsStr` slice.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push<T: AsRef<OsStr>>(&mut self, s: T) {
        self.inner.push_slice(&s.as_ref().inner)
    }

    /// Constructs an `OsString` from potentially ill-formed utf-8.
    ///
    /// It is recommended to use this function only for interoperability
    /// with existing code-bases which store paths as arbitrary `u8`
    /// sequences.
    ///
    /// Since not all platforms store paths as arbitrary `u8` sequences,
    /// this will perform a best-effort conversion. As a result, this
    /// is only guaranteed to be reversible for valid utf-8 inputs.
    ///
    /// # Platform behavior
    ///
    /// On Unix systems, any `u8` sequence can be exactly represented
    /// by an `OsString`.
    ///
    /// On Windows systems, invalid data will be replaced
    /// with U+FFFD REPLACEMENT CHARACTER.
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn from_ill_formed_utf8<'a>(bytes: &'a [u8]) -> Cow<'a, OsStr> {
        #[cfg(unix)]
        fn from_ill_formed_utf8_inner<'a>(bytes: &'a [u8]) -> Cow<'a, OsStr> {
            use os::unix::ffi::OsStrExt;
            Cow::Borrowed(OsStrExt::from_bytes(bytes))
        }

        #[cfg(windows)]
        fn from_ill_formed_utf8_inner<'a>(bytes: &'a [u8]) -> Cow<'a, OsStr> {
            match String::from_utf8_lossy(bytes) {
                Cow::Borrowed(b) => Cow::Borrowed(OsStr::new(b)),
                Cow::Owned(b) => Cow::Owned(OsString::from(b))
            }
        }

        from_ill_formed_utf8_inner(bytes)
    }

    /// Constructs an `OsString` from potentially ill-formed utf-16.
    ///
    /// Since not all platforms store paths as arbitrary `u16` sequences,
    /// this will perform a best-effort conversion. As a result, this
    /// is only guaranteed to be reversible for valid utf-16 inputs.
    ///
    /// It is recommended to use this function only for interoperability
    /// with existing code-bases which store paths as arbitrary `u16`
    /// sequences.
    ///
    /// # Platform behavior
    ///
    /// On Windows systems, any `u16` sequence can be exactly represented
    /// by an `OsString`.
    ///
    /// On Unix systems, invalid data will be replaced
    /// with U+FFFD REPLACEMENT CHARACTER.
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn from_ill_formed_utf16(wide: &[u16]) -> OsString {
        #[cfg(unix)]
        fn from_ill_formed_utf16_inner(wide: &[u16]) -> OsString {
            OsString::from(String::from_utf16_lossy(wide))
        }

        #[cfg(windows)]
        fn from_ill_formed_utf16_inner(wide: &[u16]) -> OsString {
            use os::windows::ffi::OsStringExt;
            OsStringExt::from_wide(wide)
        }

        from_ill_formed_utf16_inner(wide)
    }

}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for OsString {
    fn from(s: String) -> OsString {
        OsString { inner: Buf::from_string(s) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T: ?Sized + AsRef<OsStr>> From<&'a T> for OsString {
    fn from(s: &'a T) -> OsString {
        s.as_ref().to_os_string()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Index<ops::RangeFull> for OsString {
    type Output = OsStr;

    #[inline]
    fn index(&self, _index: ops::RangeFull) -> &OsStr {
        OsStr::from_inner(self.inner.as_slice())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ops::Deref for OsString {
    type Target = OsStr;

    #[inline]
    fn deref(&self) -> &OsStr {
        &self[..]
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for OsString {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(&**self, formatter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for OsString {
    fn eq(&self, other: &OsString) -> bool {
        &**self == &**other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<str> for OsString {
    fn eq(&self, other: &str) -> bool {
        &**self == other
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<OsString> for str {
    fn eq(&self, other: &OsString) -> bool {
        &**other == self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for OsString {}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for OsString {
    #[inline]
    fn partial_cmp(&self, other: &OsString) -> Option<cmp::Ordering> {
        (&**self).partial_cmp(&**other)
    }
    #[inline]
    fn lt(&self, other: &OsString) -> bool { &**self < &**other }
    #[inline]
    fn le(&self, other: &OsString) -> bool { &**self <= &**other }
    #[inline]
    fn gt(&self, other: &OsString) -> bool { &**self > &**other }
    #[inline]
    fn ge(&self, other: &OsString) -> bool { &**self >= &**other }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd<str> for OsString {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        (&**self).partial_cmp(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for OsString {
    #[inline]
    fn cmp(&self, other: &OsString) -> cmp::Ordering {
        (&**self).cmp(&**other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsString {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&**self).hash(state)
    }
}

impl OsStr {
    /// Coerces into an `OsStr` slice.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<S: AsRef<OsStr> + ?Sized>(s: &S) -> &OsStr {
        s.as_ref()
    }

    fn from_inner(inner: &Slice) -> &OsStr {
        unsafe { mem::transmute(inner) }
    }

    /// Yields a `&str` slice if the `OsStr` is valid unicode.
    ///
    /// This conversion may entail doing a check for UTF-8 validity.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
    }

    /// Converts an `OsStr` to a `Cow<str>`.
    ///
    /// Any non-Unicode sequences are replaced with U+FFFD REPLACEMENT CHARACTER.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_string_lossy(&self) -> Cow<str> {
        self.inner.to_string_lossy()
    }

    /// Copies the slice into an owned `OsString`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn to_os_string(&self) -> OsString {
        OsString { inner: self.inner.to_owned() }
    }

    /// Yields this `OsStr` as a byte slice.
    ///
    /// # Platform behavior
    ///
    /// On Unix systems, this is a no-op.
    ///
    /// On Windows systems, this returns `None` unless the `OsStr` is
    /// valid unicode, in which case it produces UTF-8-encoded
    /// data. This may entail checking validity.
    #[deprecated(reason = "use os_str.to_ill_formed_utf8() instead", since = "1.2.0")]
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn to_bytes(&self) -> Option<&[u8]> {
        if cfg!(windows) {
            self.to_str().map(|s| s.as_bytes())
        } else {
            Some(self.bytes())
        }
    }

    /// Creates a `CString` containing this `OsStr` data.
    ///
    /// Fails if the `OsStr` contains interior nulls.
    ///
    /// This is a convenience for creating a `CString` from
    /// `self.to_bytes()`, and inherits the platform behavior of the
    /// `to_bytes` method.
    #[deprecated(reason = "use CString::new(&*os_str.to_ill_formed_utf8()) instead", since = "1.2.0")]
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn to_cstring(&self) -> Option<CString> {
        self.to_bytes().and_then(|b| CString::new(b).ok())
    }

    /// Converts an `OsStr` to potentially ill-formed utf-8.
    ///
    /// It is recommended to use this function only for interoperability
    /// with existing code-bases which store paths as arbitrary `u8`
    /// sequences.
    ///
    /// Since not all platforms store paths as arbitrary `u8` sequences,
    /// this will perform a best-effort conversion. As a result, this
    /// is only guaranteed to be reversible for `OsStr`s consisting
    /// of valid unicode.
    ///
    /// # Platform behavior
    ///
    /// On Unix systems, any `OsString` can be exactly represented
    /// by a `u8` sequence.
    ///
    /// On Windows systems, invalid data will be replaced
    /// with U+FFFD REPLACEMENT CHARACTER.
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn to_ill_formed_utf8<'a>(&'a self) -> Cow<'a, [u8]> {
        if cfg!(windows) {
            match self.to_string_lossy() {
                Cow::Borrowed(b) => Cow::Borrowed(b.as_bytes()),
                Cow::Owned(b) => Cow::Owned(b.into_bytes())
            }
        } else {
            Cow::Borrowed(self.bytes())
        }
    }

    /// Converts an `OsStr` to potentially ill-formed utf-16.
    ///
    /// It is recommended to use this function only for interoperability
    /// with existing code-bases which store paths as arbitrary `u16`
    /// sequences.
    ///
    /// Since not all platforms store paths as arbitrary `u16` sequences,
    /// this will perform a best-effort conversion. As a result, this
    /// is only guaranteed to be reversible for `OsStr`s consisting
    /// of valid unicode.
    ///
    /// # Platform behavior
    ///
    /// On Windows systems, any `OsString` can be exactly represented
    /// by a `u16` sequence.
    ///
    /// On Unix systems, invalid data will be replaced
    /// with U+FFFD REPLACEMENT CHARACTER.
    #[unstable(feature = "convert", reason = "recently added")]
    pub fn to_ill_formed_utf16(&self) -> Vec<u16> {
        if cfg!(windows) {
            use os::windows::ffi::OsStrExt;
            self.encode_wide().collect()
        } else {
            self.to_string_lossy().utf16_units().collect()
        }
    }

    /// Gets the underlying byte representation.
    ///
    /// Note: it is *crucial* that this API is private, to avoid
    /// revealing the internal, platform-specific encodings.
    fn bytes(&self) -> &[u8] {
        unsafe { mem::transmute(&self.inner) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for OsStr {
    fn eq(&self, other: &OsStr) -> bool {
        self.bytes().eq(other.bytes())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<str> for OsStr {
    fn eq(&self, other: &str) -> bool {
        *self == *OsStr::new(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq<OsStr> for str {
    fn eq(&self, other: &OsStr) -> bool {
        *other == *OsStr::new(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Eq for OsStr {}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd for OsStr {
    #[inline]
    fn partial_cmp(&self, other: &OsStr) -> Option<cmp::Ordering> {
        self.bytes().partial_cmp(other.bytes())
    }
    #[inline]
    fn lt(&self, other: &OsStr) -> bool { self.bytes().lt(other.bytes()) }
    #[inline]
    fn le(&self, other: &OsStr) -> bool { self.bytes().le(other.bytes()) }
    #[inline]
    fn gt(&self, other: &OsStr) -> bool { self.bytes().gt(other.bytes()) }
    #[inline]
    fn ge(&self, other: &OsStr) -> bool { self.bytes().ge(other.bytes()) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialOrd<str> for OsStr {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        self.partial_cmp(OsStr::new(other))
    }
}

// FIXME (#19470): cannot provide PartialOrd<OsStr> for str until we
// have more flexible coherence rules.

#[stable(feature = "rust1", since = "1.0.0")]
impl Ord for OsStr {
    #[inline]
    fn cmp(&self, other: &OsStr) -> cmp::Ordering { self.bytes().cmp(other.bytes()) }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes().hash(state)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Debug for OsStr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.inner.fmt(formatter)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Borrow<OsStr> for OsString {
    fn borrow(&self) -> &OsStr { &self[..] }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ToOwned for OsStr {
    type Owned = OsString;
    fn to_owned(&self) -> OsString { self.to_os_string() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for OsStr {
    fn as_ref(&self) -> &OsStr {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for OsString {
    fn as_ref(&self) -> &OsStr {
        self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for str {
    fn as_ref(&self) -> &OsStr {
        OsStr::from_inner(Slice::from_str(self))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<OsStr> for String {
    fn as_ref(&self) -> &OsStr {
        (&**self).as_ref()
    }
}

impl FromInner<Buf> for OsString {
    fn from_inner(buf: Buf) -> OsString {
        OsString { inner: buf }
    }
}

impl IntoInner<Buf> for OsString {
    fn into_inner(self) -> Buf {
        self.inner
    }
}

impl AsInner<Slice> for OsStr {
    fn as_inner(&self) -> &Slice {
        &self.inner
    }
}
