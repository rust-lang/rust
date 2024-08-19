#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

//! The underlying OsString/OsStr implementation on Unix and many other
//! systems: just a `Vec<u8>`/`[u8]`.

use core::ffi::os_str::Slice;
use core::{fmt, mem, str};

use crate::borrow::{Cow, ToOwned};
use crate::boxed::Box;
use crate::collections::TryReserveError;
use crate::rc::Rc;
use crate::string::String;
use crate::sync::Arc;
use crate::vec::Vec;

#[cfg(test)]
mod tests;

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[derive(Hash)]
#[repr(transparent)]
pub struct Buf {
    pub inner: Vec<u8>,
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

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl Clone for Buf {
    #[inline]
    fn clone(&self) -> Self {
        Buf { inner: self.inner.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner)
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl Into<Vec<u8>> for Buf {
    fn into(self) -> Vec<u8> {
        self.inner
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl AsRef<[u8]> for Buf {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.inner
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
        self.inner
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: Vec<u8>) -> Self {
        Self { inner: s }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn from_string(s: String) -> Buf {
        Buf { inner: s.into_bytes() }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf { inner: Vec::with_capacity(capacity) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
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
    pub fn as_slice(&self) -> &Slice {
        // SAFETY: Slice just wraps [u8],
        // and &*self.inner is &[u8], therefore
        // transmuting &[u8] to &Slice is safe.
        unsafe { mem::transmute(&*self.inner) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut Slice {
        // SAFETY: Slice just wraps [u8],
        // and &mut *self.inner is &mut [u8], therefore
        // transmuting &mut [u8] to &mut Slice is safe.
        unsafe { mem::transmute(&mut *self.inner) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn into_string(self) -> Result<String, Buf> {
        String::from_utf8(self.inner).map_err(|p| Buf { inner: p.into_bytes() })
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.extend_from_slice(&s.inner)
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
        unsafe { mem::transmute(self.inner.into_boxed_slice()) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<[u8]> = unsafe { mem::transmute(boxed) };
        Buf { inner: inner.into_vec() }
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
        String::from_utf8_lossy(&self.inner)
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_vec() }
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
        let boxed: Box<[u8]> = self.inner.into();
        unsafe { mem::transmute(boxed) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn empty_box() -> Box<Slice> {
        let boxed: Box<[u8]> = Default::default();
        unsafe { mem::transmute(boxed) }
    }

    #[rustc_allow_incoherent_impl]
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc: Arc<[u8]> = Arc::from(&self.inner);
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
        let rc: Rc<[u8]> = Rc::from(&self.inner);
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
