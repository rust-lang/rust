//! The underlying OsString/OsStr implementation on Windows is a
//! wrapper around the "WTF-8" encoding; see the `wtf8` module for more.
use core::clone::CloneToUninit;

use crate::borrow::Cow;
use crate::collections::TryReserveError;
use crate::rc::Rc;
use crate::sync::Arc;
use crate::sys_common::wtf8::{Wtf8, Wtf8Buf, check_utf8_boundary};
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, mem};

#[derive(Clone, Hash)]
pub struct Buf {
    pub inner: Wtf8Buf,
}

impl IntoInner<Wtf8Buf> for Buf {
    fn into_inner(self) -> Wtf8Buf {
        self.inner
    }
}

impl FromInner<Wtf8Buf> for Buf {
    fn from_inner(inner: Wtf8Buf) -> Self {
        Buf { inner }
    }
}

impl AsInner<Wtf8> for Buf {
    #[inline]
    fn as_inner(&self) -> &Wtf8 {
        &self.inner
    }
}

impl fmt::Debug for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), formatter)
    }
}

impl fmt::Display for Buf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), formatter)
    }
}

#[repr(transparent)]
pub struct Slice {
    pub inner: Wtf8,
}

impl fmt::Debug for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl Buf {
    #[inline]
    pub fn into_encoded_bytes(self) -> Vec<u8> {
        self.inner.into_bytes()
    }

    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: Vec<u8>) -> Self {
        unsafe { Self { inner: Wtf8Buf::from_bytes_unchecked(s) } }
    }

    pub fn with_capacity(capacity: usize) -> Buf {
        Buf { inner: Wtf8Buf::with_capacity(capacity) }
    }

    pub fn clear(&mut self) {
        self.inner.clear()
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    pub fn from_string(s: String) -> Buf {
        Buf { inner: Wtf8Buf::from_string(s) }
    }

    pub fn as_slice(&self) -> &Slice {
        // SAFETY: Slice is just a wrapper for Wtf8,
        // and self.inner.as_slice() returns &Wtf8.
        // Therefore, transmuting &Wtf8 to &Slice is safe.
        unsafe { mem::transmute(self.inner.as_slice()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut Slice {
        // SAFETY: Slice is just a wrapper for Wtf8,
        // and self.inner.as_mut_slice() returns &mut Wtf8.
        // Therefore, transmuting &mut Wtf8 to &mut Slice is safe.
        // Additionally, care should be taken to ensure the slice
        // is always valid Wtf8.
        unsafe { mem::transmute(self.inner.as_mut_slice()) }
    }

    pub fn into_string(self) -> Result<String, Buf> {
        self.inner.into_string().map_err(|buf| Buf { inner: buf })
    }

    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.push_wtf8(&s.inner)
    }

    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    #[inline]
    pub fn leak<'a>(self) -> &'a mut Slice {
        unsafe { mem::transmute(self.inner.leak()) }
    }

    #[inline]
    pub fn into_box(self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_box()) }
    }

    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<Wtf8> = unsafe { mem::transmute(boxed) };
        Buf { inner: Wtf8Buf::from_box(inner) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        self.as_slice().into_arc()
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        self.as_slice().into_rc()
    }

    /// Provides plumbing to core `Vec::truncate`.
    /// More well behaving alternative to allowing outer types
    /// full mutable access to the core `Vec`.
    #[inline]
    pub(crate) fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Provides plumbing to core `Vec::extend_from_slice`.
    /// More well behaving alternative to allowing outer types
    /// full mutable access to the core `Vec`.
    #[inline]
    pub(crate) fn extend_from_slice(&mut self, other: &[u8]) {
        self.inner.extend_from_slice(other);
    }
}

impl Slice {
    #[inline]
    pub fn as_encoded_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: &[u8]) -> &Slice {
        unsafe { mem::transmute(Wtf8::from_bytes_unchecked(s)) }
    }

    #[track_caller]
    pub fn check_public_boundary(&self, index: usize) {
        check_utf8_boundary(&self.inner, index);
    }

    #[inline]
    pub fn from_str(s: &str) -> &Slice {
        unsafe { mem::transmute(Wtf8::from_str(s)) }
    }

    pub fn to_str(&self) -> Result<&str, crate::str::Utf8Error> {
        self.inner.as_str()
    }

    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        self.inner.to_string_lossy()
    }

    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_owned() }
    }

    pub fn clone_into(&self, buf: &mut Buf) {
        self.inner.clone_into(&mut buf.inner)
    }

    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_box()) }
    }

    pub fn empty_box() -> Box<Slice> {
        unsafe { mem::transmute(Wtf8::empty_box()) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc = self.inner.into_arc();
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Slice) }
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        let rc = self.inner.into_rc();
        unsafe { Rc::from_raw(Rc::into_raw(rc) as *const Slice) }
    }

    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        self.inner.make_ascii_lowercase()
    }

    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        self.inner.make_ascii_uppercase()
    }

    #[inline]
    pub fn to_ascii_lowercase(&self) -> Buf {
        Buf { inner: self.inner.to_ascii_lowercase() }
    }

    #[inline]
    pub fn to_ascii_uppercase(&self) -> Buf {
        Buf { inner: self.inner.to_ascii_uppercase() }
    }

    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.inner.is_ascii()
    }

    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &Self) -> bool {
        self.inner.eq_ignore_ascii_case(&other.inner)
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for Slice {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut u8) {
        // SAFETY: we're just a transparent wrapper around Wtf8
        unsafe { self.inner.clone_to_uninit(dst) }
    }
}
