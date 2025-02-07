//! The underlying OsString/OsStr implementation on Unix and many other
//! systems: just a `Vec<u8>`/`[u8]`.

use core::clone::CloneToUninit;
use core::str::advance_utf8;

use crate::borrow::Cow;
use crate::collections::TryReserveError;
use crate::fmt::Write;
use crate::rc::Rc;
use crate::sync::Arc;
use crate::sys_common::{AsInner, FromInner, IntoInner};
use crate::{fmt, mem, str};

#[cfg(test)]
mod tests;

#[derive(Hash)]
#[repr(transparent)]
pub struct Buf {
    pub inner: Vec<u8>,
}

#[repr(transparent)]
pub struct Slice {
    pub inner: [u8],
}

impl IntoInner<Vec<u8>> for Buf {
    fn into_inner(self) -> Vec<u8> {
        self.inner
    }
}

impl FromInner<Vec<u8>> for Buf {
    fn from_inner(inner: Vec<u8>) -> Self {
        Buf { inner }
    }
}

impl AsInner<[u8]> for Buf {
    #[inline]
    fn as_inner(&self) -> &[u8] {
        &self.inner
    }
}

impl fmt::Debug for Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl fmt::Display for Buf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), f)
    }
}

impl fmt::Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner.utf8_chunks().debug(), f)
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Corresponds to `Formatter::pad`, but for `OsStr` instead of `str`.

        // Make sure there's a fast path up front.
        if f.options().get_width().is_none() && f.options().get_precision().is_none() {
            return self.write_lossy(f);
        }

        // The `precision` field can be interpreted as a maximum width for the
        // string being formatted.
        let max_char_count = f.options().get_precision().unwrap_or(usize::MAX);
        let (truncated, char_count) = truncate_chars(&self.inner, max_char_count);

        // If our string is longer than the maximum width, truncate it and
        // handle other flags in terms of the truncated string.
        // SAFETY: The truncation splits at Unicode scalar value boundaries.
        let s = unsafe { Slice::from_encoded_bytes_unchecked(truncated) };

        // The `width` field is more of a minimum width parameter at this point.
        if let Some(width) = f.options().get_width()
            && char_count < width
        {
            // If we're under the minimum width, then fill up the minimum width
            // with the specified string + some alignment.
            let post_padding = f.padding(width - char_count, fmt::Alignment::Left)?;
            s.write_lossy(f)?;
            post_padding.write(f)
        } else {
            // If we're over the minimum width or there is no minimum width, we
            // can just emit the string.
            s.write_lossy(f)
        }
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
        self.inner
    }

    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: Vec<u8>) -> Self {
        Self { inner: s }
    }

    #[inline]
    pub fn into_string(self) -> Result<String, Buf> {
        String::from_utf8(self.inner).map_err(|p| Buf { inner: p.into_bytes() })
    }

    #[inline]
    pub fn from_string(s: String) -> Buf {
        Buf { inner: s.into_bytes() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Buf {
        Buf { inner: Vec::with_capacity(capacity) }
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
        self.inner.extend_from_slice(&s.inner)
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
        // SAFETY: Slice just wraps [u8],
        // and &*self.inner is &[u8], therefore
        // transmuting &[u8] to &Slice is safe.
        unsafe { mem::transmute(self.inner.as_slice()) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut Slice {
        // SAFETY: Slice just wraps [u8],
        // and &mut *self.inner is &mut [u8], therefore
        // transmuting &mut [u8] to &mut Slice is safe.
        unsafe { mem::transmute(self.inner.as_mut_slice()) }
    }

    #[inline]
    pub fn leak<'a>(self) -> &'a mut Slice {
        unsafe { mem::transmute(self.inner.leak()) }
    }

    #[inline]
    pub fn into_box(self) -> Box<Slice> {
        unsafe { mem::transmute(self.inner.into_boxed_slice()) }
    }

    #[inline]
    pub fn from_box(boxed: Box<Slice>) -> Buf {
        let inner: Box<[u8]> = unsafe { mem::transmute(boxed) };
        Buf { inner: inner.into_vec() }
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
        &self.inner
    }

    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: &[u8]) -> &Slice {
        unsafe { mem::transmute(s) }
    }

    #[track_caller]
    #[inline]
    pub fn check_public_boundary(&self, index: usize) {
        if index == 0 || index == self.inner.len() {
            return;
        }
        if index < self.inner.len()
            && (self.inner[index - 1].is_ascii() || self.inner[index].is_ascii())
        {
            return;
        }

        slow_path(&self.inner, index);

        /// We're betting that typical splits will involve an ASCII character.
        ///
        /// Putting the expensive checks in a separate function generates notably
        /// better assembly.
        #[track_caller]
        #[inline(never)]
        fn slow_path(bytes: &[u8], index: usize) {
            let (before, after) = bytes.split_at(index);

            // UTF-8 takes at most 4 bytes per codepoint, so we don't
            // need to check more than that.
            let after = after.get(..4).unwrap_or(after);
            match str::from_utf8(after) {
                Ok(_) => return,
                Err(err) if err.valid_up_to() != 0 => return,
                Err(_) => (),
            }

            for len in 2..=4.min(index) {
                let before = &before[index - len..];
                if str::from_utf8(before).is_ok() {
                    return;
                }
            }

            panic!("byte index {index} is not an OsStr boundary");
        }
    }

    #[inline]
    pub fn from_str(s: &str) -> &Slice {
        unsafe { Slice::from_encoded_bytes_unchecked(s.as_bytes()) }
    }

    #[inline]
    pub fn to_str(&self) -> Result<&str, crate::str::Utf8Error> {
        str::from_utf8(&self.inner)
    }

    #[inline]
    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.inner)
    }

    /// Writes the string as lossy UTF-8 like [`String::from_utf8_lossy`].
    /// It ignores formatter flags.
    fn write_lossy(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for chunk in self.inner.utf8_chunks() {
            f.write_str(chunk.valid())?;
            if !chunk.invalid().is_empty() {
                f.write_char(char::REPLACEMENT_CHARACTER)?;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_vec() }
    }

    #[inline]
    pub fn clone_into(&self, buf: &mut Buf) {
        self.inner.clone_into(&mut buf.inner)
    }

    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        let boxed: Box<[u8]> = self.inner.into();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn empty_box() -> Box<Slice> {
        let boxed: Box<[u8]> = Default::default();
        unsafe { mem::transmute(boxed) }
    }

    #[inline]
    pub fn into_arc(&self) -> Arc<Slice> {
        let arc: Arc<[u8]> = Arc::from(&self.inner);
        unsafe { Arc::from_raw(Arc::into_raw(arc) as *const Slice) }
    }

    #[inline]
    pub fn into_rc(&self) -> Rc<Slice> {
        let rc: Rc<[u8]> = Rc::from(&self.inner);
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

/// Counts the number of Unicode scalar values in the byte string, allowing
/// invalid UTF-8 sequences. For invalid sequences, the maximal prefix of a
/// valid UTF-8 code unit counts as one. Only up to `max_chars` scalar values
/// are scanned. Returns the character count and the byte length.
fn truncate_chars(bytes: &[u8], max_chars: usize) -> (&[u8], usize) {
    let mut iter = bytes.iter();
    let mut char_count = 0;
    while !iter.is_empty() && char_count < max_chars {
        advance_utf8(&mut iter);
        char_count += 1;
    }
    let byte_len = bytes.len() - iter.len();
    let truncated = unsafe { bytes.get_unchecked(..byte_len) };
    (truncated, char_count)
}
