#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

//! The underlying OsString/OsStr implementation on Unix and many other
//! systems: just a `Vec<u8>`/`[u8]`.

use crate::clone::CloneToUninit;
use crate::fmt::Write;
use crate::ptr::addr_of_mut;
use crate::{fmt, mem, str};

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
#[repr(transparent)]
#[rustc_has_incoherent_inherent_impls]
pub struct Slice {
    pub inner: [u8],
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
impl fmt::Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner.utf8_chunks().debug(), f)
    }
}

#[unstable(
    feature = "os_str_internals",
    reason = "internal details of the implementation of os str",
    issue = "none"
)]
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

impl Slice {
    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn as_encoded_bytes(&self) -> &[u8] {
        &self.inner
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub unsafe fn from_encoded_bytes_unchecked(s: &[u8]) -> &Slice {
        // SAFETY: Slice is just a wrapper of [u8]
        unsafe { mem::transmute(s) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
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

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    #[inline]
    pub fn from_str(s: &str) -> &Slice {
        // SAFETY: Slice is just a wrapper of [u8]
        unsafe { Slice::from_encoded_bytes_unchecked(s.as_bytes()) }
    }

    #[unstable(
        feature = "os_str_internals",
        reason = "internal details of the implementation of os str",
        issue = "none"
    )]
    pub fn to_str(&self) -> Result<&str, crate::str::Utf8Error> {
        str::from_utf8(&self.inner)
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
        // SAFETY: we're just a wrapper around [u8]
        unsafe { self.inner.clone_to_uninit(addr_of_mut!((*dst).inner)) }
    }
}
