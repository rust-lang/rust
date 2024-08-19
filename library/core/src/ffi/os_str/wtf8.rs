#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

//! The underlying OsString/OsStr implementation on Windows is a
//! wrapper around the "WTF-8" encoding; see the `wtf8` module for more.
use crate::clone::CloneToUninit;
use crate::ffi::wtf8::{check_utf8_boundary, Wtf8};
use crate::ptr::addr_of_mut;
use crate::{fmt, mem};

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[repr(transparent)]
#[rustc_has_incoherent_inherent_impls]
pub struct Slice {
    pub inner: Wtf8,
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl fmt::Debug for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, formatter)
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl fmt::Display for Slice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl Slice {
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn as_encoded_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: &[u8]) -> &Slice {
        // SAFETY:: Slice is just a wrapper of Wtf8
        unsafe { mem::transmute(Wtf8::from_bytes_unchecked(s)) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[track_caller]
    pub fn check_public_boundary(&self, index: usize) {
        check_utf8_boundary(&self.inner, index);
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn from_str(s: &str) -> &Slice {
        // SAFETY: Slice is just a wrapper of wtf8
        unsafe { mem::transmute(Wtf8::from_str(s)) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn to_str(&self) -> Result<&str, crate::str::Utf8Error> {
        self.inner.as_str()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        self.inner.make_ascii_lowercase()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        self.inner.make_ascii_uppercase()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        self.inner.is_ascii()
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &Self) -> bool {
        self.inner.eq_ignore_ascii_case(&other.inner)
    }
}

#[unstable(feature = "clone_to_uninit", issue = "126799")]
unsafe impl CloneToUninit for Slice {
    #[inline]
    #[cfg_attr(debug_assertions, track_caller)]
    unsafe fn clone_to_uninit(&self, dst: *mut Self) {
        // SAFETY: we're just a wrapper around Wtf8
        unsafe { self.inner.clone_to_uninit(addr_of_mut!((*dst).inner)) }
    }
}
