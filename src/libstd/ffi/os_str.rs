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

#![unstable(feature = "os",
            reason = "recently added as part of path/io reform")]

use core::prelude::*;

use borrow::{Borrow, ToOwned};
use fmt::{self, Debug};
use mem;
use string::{String, CowString};
use ops;
use cmp;
use hash::{Hash, Hasher};
#[cfg(stage0)] use hash::Writer;
use old_path::{Path, GenericPath};

use sys::os_str::{Buf, Slice};
use sys_common::{AsInner, IntoInner, FromInner};
use super::AsOsStr;

/// Owned, mutable OS strings.
#[derive(Clone)]
pub struct OsString {
    inner: Buf
}

/// Slices into OS strings.
pub struct OsStr {
    inner: Slice
}

impl OsString {
    /// Constructs an `OsString` at no cost by consuming a `String`.
    pub fn from_string(s: String) -> OsString {
        OsString { inner: Buf::from_string(s) }
    }

    /// Constructs an `OsString` by copying from a `&str` slice.
    ///
    /// Equivalent to: `OsString::from_string(String::from_str(s))`.
    pub fn from_str(s: &str) -> OsString {
        OsString { inner: Buf::from_str(s) }
    }

    /// Constructs a new empty `OsString`.
    pub fn new() -> OsString {
        OsString { inner: Buf::from_string(String::new()) }
    }

    /// Convert the `OsString` into a `String` if it contains valid Unicode data.
    ///
    /// On failure, ownership of the original `OsString` is returned.
    pub fn into_string(self) -> Result<String, OsString> {
        self.inner.into_string().map_err(|buf| OsString { inner: buf} )
    }

    /// Extend the string with the given `&OsStr` slice.
    pub fn push_os_str(&mut self, s: &OsStr) {
        self.inner.push_slice(&s.inner)
    }
}

impl ops::Index<ops::RangeFull> for OsString {
    type Output = OsStr;

    #[inline]
    fn index(&self, _index: &ops::RangeFull) -> &OsStr {
        unsafe { mem::transmute(self.inner.as_slice()) }
    }
}

impl ops::Deref for OsString {
    type Target = OsStr;

    #[inline]
    fn deref(&self) -> &OsStr {
        &self[..]
    }
}

impl Debug for OsString {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Debug::fmt(&**self, formatter)
    }
}

impl PartialEq for OsString {
    fn eq(&self, other: &OsString) -> bool {
        &**self == &**other
    }
}

impl PartialEq<str> for OsString {
    fn eq(&self, other: &str) -> bool {
        &**self == other
    }
}

impl PartialEq<OsString> for str {
    fn eq(&self, other: &OsString) -> bool {
        &**other == self
    }
}

impl Eq for OsString {}

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

impl PartialOrd<str> for OsString {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        (&**self).partial_cmp(other)
    }
}

impl Ord for OsString {
    #[inline]
    fn cmp(&self, other: &OsString) -> cmp::Ordering {
        (&**self).cmp(&**other)
    }
}

#[cfg(stage0)]
impl<'a, S: Hasher + Writer> Hash<S> for OsString {
    #[inline]
    fn hash(&self, state: &mut S) {
        (&**self).hash(state)
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsString {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&**self).hash(state)
    }
}

impl OsStr {
    /// Coerce directly from a `&str` slice to a `&OsStr` slice.
    pub fn from_str(s: &str) -> &OsStr {
        unsafe { mem::transmute(Slice::from_str(s)) }
    }

    /// Yield a `&str` slice if the `OsStr` is valid unicode.
    ///
    /// This conversion may entail doing a check for UTF-8 validity.
    pub fn to_str(&self) -> Option<&str> {
        self.inner.to_str()
    }

    /// Convert an `OsStr` to a `CowString`.
    ///
    /// Any non-Unicode sequences are replaced with U+FFFD REPLACEMENT CHARACTER.
    pub fn to_string_lossy(&self) -> CowString {
        self.inner.to_string_lossy()
    }

    /// Copy the slice into an owned `OsString`.
    pub fn to_os_string(&self) -> OsString {
        OsString { inner: self.inner.to_owned() }
    }

    /// Get the underlying byte representation.
    ///
    /// Note: it is *crucial* that this API is private, to avoid
    /// revealing the internal, platform-specific encodings.
    fn bytes(&self) -> &[u8] {
        unsafe { mem::transmute(&self.inner) }
    }
}

impl PartialEq for OsStr {
    fn eq(&self, other: &OsStr) -> bool {
        self.bytes().eq(other.bytes())
    }
}

impl PartialEq<str> for OsStr {
    fn eq(&self, other: &str) -> bool {
        *self == *OsStr::from_str(other)
    }
}

impl PartialEq<OsStr> for str {
    fn eq(&self, other: &OsStr) -> bool {
        *other == *OsStr::from_str(self)
    }
}

impl Eq for OsStr {}

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

impl PartialOrd<str> for OsStr {
    #[inline]
    fn partial_cmp(&self, other: &str) -> Option<cmp::Ordering> {
        self.partial_cmp(OsStr::from_str(other))
    }
}

// FIXME (#19470): cannot provide PartialOrd<OsStr> for str until we
// have more flexible coherence rules.

impl Ord for OsStr {
    #[inline]
    fn cmp(&self, other: &OsStr) -> cmp::Ordering { self.bytes().cmp(other.bytes()) }
}

#[cfg(stage0)]
impl<'a, S: Hasher + Writer> Hash<S> for OsStr {
    #[inline]
    fn hash(&self, state: &mut S) {
        self.bytes().hash(state)
    }
}
#[cfg(not(stage0))]
#[stable(feature = "rust1", since = "1.0.0")]
impl Hash for OsStr {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bytes().hash(state)
    }
}

impl Debug for OsStr {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.inner.fmt(formatter)
    }
}

impl Borrow<OsStr> for OsString {
    fn borrow(&self) -> &OsStr { &self[..] }
}

impl ToOwned for OsStr {
    type Owned = OsString;
    fn to_owned(&self) -> OsString { self.to_os_string() }
}

impl<'a, T: AsOsStr + ?Sized> AsOsStr for &'a T {
    fn as_os_str(&self) -> &OsStr {
        (*self).as_os_str()
    }
}

impl AsOsStr for OsStr {
    fn as_os_str(&self) -> &OsStr {
        self
    }
}

impl AsOsStr for OsString {
    fn as_os_str(&self) -> &OsStr {
        &self[..]
    }
}

impl AsOsStr for str {
    fn as_os_str(&self) -> &OsStr {
        OsStr::from_str(self)
    }
}

impl AsOsStr for String {
    fn as_os_str(&self) -> &OsStr {
        OsStr::from_str(&self[..])
    }
}

#[cfg(unix)]
impl AsOsStr for Path {
    fn as_os_str(&self) -> &OsStr {
        unsafe { mem::transmute(self.as_vec()) }
    }
}

#[cfg(windows)]
impl AsOsStr for Path {
    fn as_os_str(&self) -> &OsStr {
        // currently .as_str() is actually infallible on windows
        OsStr::from_str(self.as_str().unwrap())
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
