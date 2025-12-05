//! An OsString/OsStr implementation that is guaranteed to be UTF-8.

use core::clone::CloneToUninit;

use crate::borrow::Cow;
use crate::collections::TryReserveError;
use crate::rc::Rc;
use crate::sync::Arc;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, mem};

#[derive(Hash)]
#[repr(transparent)]
pub struct Buf {
    pub inner: String,
}

#[repr(transparent)]
pub struct Slice {
    pub inner: str,
}

impl IntoInner<String> for Buf {
    fn into_inner(self) -> String {
        self.inner
    }
}

impl FromInner<String> for Buf {
    fn from_inner(inner: String) -> Self {
        Buf { inner }
    }
}

impl AsInner<str> for Buf {
    #[inline]
    fn as_inner(&self) -> &str {
        &self.inner
    }
}

impl fmt::Debug for Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl fmt::Display for Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

impl fmt::Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, f)
    }
}

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

impl Buf {
    #[inline]
    pub fn into_encoded_bytes(self) -> Vec<u8> {
        self.inner.into_bytes()
    }

    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: Vec<u8>) -> Self {
        unsafe { Self { inner: String::from_utf8_unchecked(s) } }
    }

    #[inline]
    pub fn into_string(self) -> Result<String, Buf> {
        Ok(self.inner)
    }

    #[inline]
    pub const fn from_string(s: String) -> Buf {
        Buf { inner: s }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf { inner: String::with_capacity(capacity) }
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
    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.push_str(&s.inner)
    }

    #[inline]
    pub fn push_str(&mut self, s: &str) {
        self.inner.push_str(s);
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional)
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve(additional)
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.inner.reserve_exact(additional)
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.inner.try_reserve_exact(additional)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit()
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.inner.shrink_to(min_capacity)
    }

    #[inline]
    pub fn as_slice(&self) -> &Slice {
        Slice::from_str(&self.inner)
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut Slice {
        Slice::from_mut_str(&mut self.inner)
    }

    #[inline]
    pub fn leak<'a>(self) -> &'a mut Slice {
        Slice::from_mut_str(self.inner.leak())
    }

    #[inline]
    pub fn into_box(self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_boxed_str()) }
    }

    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<str> = unsafe { mem::transmute(boxed) };
        Buf { inner: inner.into_string() }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        self.as_slice().into_arc()
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        self.as_slice().into_rc()
    }

    /// Provides plumbing to `Vec::truncate` without giving full mutable access
    /// to the `Vec`.
    ///
    /// # Safety
    ///
    /// The length must be at an `OsStr` boundary, according to
    /// `Slice::check_public_boundary`.
    #[inline]
    pub unsafe fn truncate_unchecked(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// Provides plumbing to `Vec::extend_from_slice` without giving full
    /// mutable access to the `Vec`.
    ///
    /// # Safety
    ///
    /// The slice must be valid for the platform encoding (as described in
    /// `OsStr::from_encoded_bytes_unchecked`). For this encoding, that means
    /// `other` must be valid UTF-8.
    #[inline]
    pub unsafe fn extend_from_slice_unchecked(&mut self, other: &[u8]) {
        self.inner.push_str(unsafe { str::from_utf8_unchecked(other) });
    }
}

impl Slice {
    #[inline]
    pub fn as_encoded_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: &[u8]) -> &Slice {
        Slice::from_str(unsafe { str::from_utf8_unchecked(s) })
    }

    #[track_caller]
    #[inline]
    pub fn check_public_boundary(&self, index: usize) {
        if !self.inner.is_char_boundary(index) {
            panic!("byte index {index} is not an OsStr boundary");
        }
    }

    #[inline]
    pub fn from_str(s: &str) -> &Slice {
        // SAFETY: Slice is just a wrapper over str.
        unsafe { mem::transmute(s) }
    }

    #[inline]
    fn from_mut_str(s: &mut str) -> &mut Slice {
        // SAFETY: Slice is just a wrapper over str.
        unsafe { mem::transmute(s) }
    }

    #[inline]
    pub fn to_str(&self) -> Result<&str, crate::str::Utf8Error> {
        Ok(&self.inner)
    }

    #[inline]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.inner)
    }

    #[inline]
    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_owned() }
    }

    #[inline]
    pub fn clone_into(&self, buf: &mut Buf) {
        self.inner.clone_into(&mut buf.inner)
    }

    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        let boxed: Box<str> = self.inner.into();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn empty_box() -> Box<Slice> {
        let boxed: Box<str> = Default::default();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc: Arc<str> = Arc::from(&self.inner);
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Slice) }
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        let rc: Rc<str> = Rc::from(&self.inner);
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
        // SAFETY: we're just a transparent wrapper around [u8]
        unsafe { self.inner.clone_to_uninit(dst) }
    }
}
