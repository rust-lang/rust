#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

//! The underlying OsString/OsStr implementation on Windows is a
//! wrapper around the "WTF-8" encoding; see the `wtf8` module for more.
use core::ffi::os_str::Slice;
use core::ffi::wtf8::Wtf8;
use core::{fmt, mem};

use crate::borrow::Cow;
use crate::boxed::Box;
use crate::collections::TryReserveError;
use crate::ffi::wtf8::Wtf8Buf;
use crate::rc::Rc;
use crate::string::String;
use crate::sync::Arc;
use crate::vec::Vec;

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[derive(Clone, Hash)]
pub struct Buf {
    pub inner: Wtf8Buf,
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl Into<Wtf8Buf> for Buf {
    fn into(self) -> Wtf8Buf {
        self.inner
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl From<Wtf8Buf> for Buf {
    fn from(inner: Wtf8Buf) -> Self {
        Buf { inner }
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl AsRef<Wtf8> for Buf {
    #[inline]
    fn as_ref(&self) -> &Wtf8 {
        &self.inner
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl fmt::Debug for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), formatter)
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl fmt::Display for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), formatter)
    }
}

impl Buf {
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_encoded_bytes(self) -> Vec<u8> {
        self.inner.into_bytes()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: Vec<u8>) -> Self {
        unsafe { Self { inner: Wtf8Buf::from_bytes_unchecked(s) } }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf { inner: Wtf8Buf::with_capacity(capacity) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn from_string(s: String) -> Buf {
        Buf { inner: Wtf8Buf::from_string(s) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn as_slice(&self) -> &Slice {
        // SAFETY: Slice is just a wrapper for Wtf8,
        // and self.inner.as_slice() returns &Wtf8.
        // Therefore, transmuting &Wtf8 to &Slice is safe.
        unsafe { mem::transmute(self.inner.as_slice()) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn as_mut_slice(&mut self) -> &mut Slice {
        // SAFETY: Slice is just a wrapper for Wtf8,
        // and self.inner.as_mut_slice() returns &mut Wtf8.
        // Therefore, transmuting &mut Wtf8 to &mut Slice is safe.
        // Additionally, care should be taken to ensure the slice
        // is always valid Wtf8.
        unsafe { mem::transmute(self.inner.as_mut_slice()) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn into_string(self) -> Result<String, Buf> {
        self.inner.into_string().map_err(|buf| Buf { inner: buf })
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.push_wtf8(&s.inner)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn leak<'a>(self) -> &'a mut Slice {
        unsafe { mem::transmute(self.inner.leak()) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_box(self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_box()) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<Wtf8> = unsafe { mem::transmute(boxed) };
        Buf { inner: Wtf8Buf::from_box(inner) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        self.as_slice().into_arc()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        self.as_slice().into_rc()
    }

    /// Provides plumbing to core `Vec::truncate`.
    /// More well behaving alternative to allowing outer types
    /// full mutable access to the core `Vec`.
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub(crate) fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Provides plumbing to core `Vec::extend_from_slice`.
    /// More well behaving alternative to allowing outer types
    /// full mutable access to the core `Vec`.
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub(crate) fn extend_from_slice(&mut self, other: &[u8]) {
        self.inner.extend_from_slice(other);
    }
}

impl Slice {
    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.inner.to_string_lossy()
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_owned() }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn clone_into(&self, buf: &mut Buf) {
        self.inner.clone_into(&mut buf.inner)
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_box()) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn empty_box() -> Box<Slice> {
        unsafe { mem::transmute(Wtf8::empty_box()) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc = self.inner.into_arc();
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Slice) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        let rc = self.inner.into_rc();
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Slice) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn to_ascii_lowercase(&self) -> Buf {
        Buf { inner: self.inner.to_ascii_lowercase() }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn to_ascii_uppercase(&self) -> Buf {
        Buf { inner: self.inner.to_ascii_uppercase() }
    }
}
