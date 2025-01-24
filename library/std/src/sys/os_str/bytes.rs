//! The underlying OsString/OsStr implementation on Unix and many other
//! systems: just a `Vec<u8>`/`[u8]`.

use core::clone::CloneToUninit;

use crate::borrow::Cow;
use crate::collections::TryReserveError;
use crate::fmt::Write;
use crate::rc::Rc;
use crate::sync::Arc;
use crate::sys_common::{AsInner, IntoInner};
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

impl fmt::Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner.utf8_chunks().debug(), f)
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If we're the empty string then our iterator won't actually yield
        // anything, so perform the formatting manually
        if self.inner.is_empty() {
            return "".fmt(f);
        }

        for chunk in self.inner.utf8_chunks() {
            let valid = chunk.valid();
            // If we successfully decoded the whole chunk as a valid string then
            // we can return a direct formatting of the string which will also
            // respect various formatting flags if possible.
            if chunk.invalid().is_empty() {
                return valid.fmt(f);
            }

            f.write_str(valid)?;
            f.write_char(char::REPLACEMENT_CHARACTER)?;
        }
        Ok(())
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

impl IntoInner<Vec<u8>> for Buf {
    fn into_inner(self) -> Vec<u8> {
        self.inner
    }
}

impl AsInner<[u8]> for Buf {
    #[inline]
    fn as_inner(&self) -> &[u8] {
        &self.inner
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
        unsafe { mem::transmute(&*self.inner) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut Slice {
        // SAFETY: Slice just wraps [u8],
        // and &mut *self.inner is &mut [u8], therefore
        // transmuting &mut [u8] to &mut Slice is safe.
        unsafe { mem::transmute(&mut *self.inner) }
    }

    pub fn into_string(self) -> Result<String, Buf> {
        String::from_utf8(self.inner).map_err(|p| Buf { inner: p.into_bytes() })
    }

    pub fn push_slice(&mut self, s: &Slice) {
        self.inner.extend_from_slice(&s.inner)
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

    pub fn to_str(&self) -> Result<&str, crate::str::Utf8Error> {
        str::from_utf8(&self.inner)
    }

    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.inner)
    }

    pub fn to_owned(&self) -> Buf {
        Buf { inner: self.inner.to_vec() }
    }

    pub fn clone_into(&self, buf: &mut Buf) {
        self.inner.clone_into(&mut buf.inner)
    }

    #[inline]
    pub fn into_box(&self) -> Box<Slice> {
        let boxed: Box<[u8]> = self.inner.into();
        unsafe { mem::transmute(boxed) }
    }

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
