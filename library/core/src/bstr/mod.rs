//! The `ByteStr` type and trait implementations.

mod traits;

#[unstable(feature = "bstr_internals", issue = "none")]
pub use traits::{impl_partial_eq, impl_partial_eq_n, impl_partial_eq_ord};

use crate::borrow::{Borrow, BorrowMut};
use crate::fmt::{self, Alignment};
use crate::ops::{Deref, DerefMut, DerefPure};

/// A wrapper for `&[u8]` representing a human-readable string that's conventionally, but not
/// always, UTF-8.
///
/// Unlike `&str`, this type permits non-UTF-8 contents, making it suitable for user input,
/// non-native filenames (as `Path` only supports native filenames), and other applications that
/// need to round-trip whatever data the user provides.
///
/// For an owned, growable byte string buffer, use
/// [`ByteString`](../../std/bstr/struct.ByteString.html).
///
/// `ByteStr` implements `Deref` to `[u8]`, so all methods available on `[u8]` are available on
/// `ByteStr`.
///
/// # Representation
///
/// A `&ByteStr` has the same representation as a `&str`. That is, a `&ByteStr` is a wide pointer
/// which includes a pointer to some bytes and a length.
///
/// # Trait implementations
///
/// The `ByteStr` type has a number of trait implementations, and in particular, defines equality
/// and comparisons between `&ByteStr`, `&str`, and `&[u8]`, for convenience.
///
/// The `Debug` implementation for `ByteStr` shows its bytes as a normal string, with invalid UTF-8
/// presented as hex escape sequences.
///
/// The `Display` implementation behaves as if the `ByteStr` were first lossily converted to a
/// `str`, with invalid UTF-8 presented as the Unicode replacement character (�).
#[unstable(feature = "bstr", issue = "134915")]
#[repr(transparent)]
#[doc(alias = "BStr")]
pub struct ByteStr(pub [u8]);

impl ByteStr {
    /// Creates a `ByteStr` slice from anything that can be converted to a byte slice.
    ///
    /// This is a zero-cost conversion.
    ///
    /// # Example
    ///
    /// You can create a `ByteStr` from a byte array, a byte slice or a string slice:
    ///
    /// ```
    /// # #![feature(bstr)]
    /// # use std::bstr::ByteStr;
    /// let a = ByteStr::new(b"abc");
    /// let b = ByteStr::new(&b"abc"[..]);
    /// let c = ByteStr::new("abc");
    ///
    /// assert_eq!(a, b);
    /// assert_eq!(a, c);
    /// ```
    #[inline]
    #[unstable(feature = "bstr", issue = "134915")]
    #[rustc_const_unstable(feature = "const_convert", issue = "143773")]
    pub const fn new<B: ?Sized + [const] AsRef<[u8]>>(bytes: &B) -> &Self {
        ByteStr::from_bytes(bytes.as_ref())
    }

    /// Returns the same string as `&ByteStr`.
    ///
    /// This method is redundant when used directly on `&ByteStr`, but
    /// it helps dereferencing other "container" types,
    /// for example `Box<ByteStr>` or `Arc<ByteStr>`.
    #[inline]
    // #[unstable(feature = "str_as_str", issue = "130366")]
    #[unstable(feature = "bstr", issue = "134915")]
    pub const fn as_byte_str(&self) -> &ByteStr {
        self
    }

    /// Returns the same string as `&mut ByteStr`.
    ///
    /// This method is redundant when used directly on `&mut ByteStr`, but
    /// it helps dereferencing other "container" types,
    /// for example `Box<ByteStr>` or `MutexGuard<ByteStr>`.
    #[inline]
    // #[unstable(feature = "str_as_str", issue = "130366")]
    #[unstable(feature = "bstr", issue = "134915")]
    pub const fn as_mut_byte_str(&mut self) -> &mut ByteStr {
        self
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr_internals", issue = "none")]
    pub const fn from_bytes(slice: &[u8]) -> &Self {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`, so we can turn a reference to
        // the wrapped type into a reference to the wrapper type.
        unsafe { &*(slice as *const [u8] as *const Self) }
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr_internals", issue = "none")]
    pub const fn from_bytes_mut(slice: &mut [u8]) -> &mut Self {
        // SAFETY: `ByteStr` is a transparent wrapper around `[u8]`, so we can turn a reference to
        // the wrapped type into a reference to the wrapper type.
        unsafe { &mut *(slice as *mut [u8] as *mut Self) }
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr_internals", issue = "none")]
    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    #[doc(hidden)]
    #[unstable(feature = "bstr_internals", issue = "none")]
    #[inline]
    #[rustc_const_unstable(feature = "bstr_internals", issue = "none")]
    pub const fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const Deref for ByteStr {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const DerefMut for ByteStr {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "deref_pure_trait", issue = "87121")]
unsafe impl DerefPure for ByteStr {}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Debug for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"")?;
        for chunk in self.utf8_chunks() {
            for c in chunk.valid().chars() {
                match c {
                    '\0' => write!(f, "\\0")?,
                    '\x01'..='\x7f' => write!(f, "{}", (c as u8).escape_ascii())?,
                    _ => write!(f, "{}", c.escape_debug())?,
                }
            }
            write!(f, "{}", chunk.invalid().escape_ascii())?;
        }
        write!(f, "\"")?;
        Ok(())
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl fmt::Display for ByteStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn emit(byte_str: &ByteStr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            for chunk in byte_str.utf8_chunks() {
                f.write_str(chunk.valid())?;
                if !chunk.invalid().is_empty() {
                    f.write_str("\u{FFFD}")?;
                }
            }

            Ok(())
        }

        let requested_width = f.width().unwrap_or(0);
        if requested_width == 0 && f.precision().is_none() {
            // Avoid counting the characters if no truncation or padding was
            // requested.
            return emit(self, f);
        }

        let (truncated, actual_width) = match f.precision() {
            // The entire string is truncated away. Weird, but ok.
            Some(0) => (ByteStr::new(&[]), 0),
            // Advance through string until we run out of space.
            Some(precision) => {
                let mut remaining_width = precision;
                let mut chunks = self.utf8_chunks();
                let mut current_width = 0;
                let mut offset = 0;
                loop {
                    let Some(chunk) = chunks.next() else {
                        // We reached the end of the string without running out
                        // of space, so print the entire string.
                        break (self, current_width);
                    };

                    let mut chars = chunk.valid().char_indices();
                    let Err(remaining) = chars.advance_by(remaining_width) else {
                        // We've counted off `precision` characters, so truncate
                        // the string at the current offset.
                        break (&self[..offset + chars.offset()], precision);
                    };

                    offset += chunk.valid().len();
                    current_width += remaining_width - remaining.get();
                    remaining_width = remaining.get();

                    // `remaining_width` cannot be zero, there is still space
                    // remaining. So next, count the � character emitted for
                    // the invalid chunk (if it exists).
                    if !chunk.invalid().is_empty() {
                        offset += chunk.invalid().len();
                        current_width += 1;
                        remaining_width -= 1;

                        if remaining_width == 0 {
                            break (&self[..offset], precision);
                        }
                    }
                }
            }
            // The string shouldn't be truncated at all, so just count the number
            // of characters to calculate the padding.
            None => {
                let actual_width = self
                    .utf8_chunks()
                    .map(|chunk| {
                        chunk.valid().chars().count()
                            + if chunk.invalid().is_empty() { 0 } else { 1 }
                    })
                    .sum();
                (self, actual_width)
            }
        };

        // The width is originally stored as a 16-bit number, so this cannot fail.
        let padding = u16::try_from(requested_width.saturating_sub(actual_width)).unwrap();

        let post_padding = f.padding(padding, Alignment::Left)?;
        emit(truncated, f)?;
        post_padding.write(f)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsRef<[u8]> for ByteStr {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsRef<ByteStr> for ByteStr {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        self
    }
}

// `impl AsRef<ByteStr> for [u8]` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsRef<ByteStr> for str {
    #[inline]
    fn as_ref(&self) -> &ByteStr {
        ByteStr::new(self)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const AsMut<[u8]> for ByteStr {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

// `impl AsMut<ByteStr> for [u8]` omitted to avoid widespread inference failures

// `impl Borrow<ByteStr> for [u8]` omitted to avoid widespread inference failures

// `impl Borrow<ByteStr> for str` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const Borrow<[u8]> for ByteStr {
    #[inline]
    fn borrow(&self) -> &[u8] {
        &self.0
    }
}

// `impl BorrowMut<ByteStr> for [u8]` omitted to avoid widespread inference failures

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl const BorrowMut<[u8]> for ByteStr {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> Default for &'a ByteStr {
    fn default() -> Self {
        ByteStr::from_bytes(b"")
    }
}

#[unstable(feature = "bstr", issue = "134915")]
impl<'a> Default for &'a mut ByteStr {
    fn default() -> Self {
        ByteStr::from_bytes_mut(&mut [])
    }
}

// Omitted due to inference failures
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a, const N: usize> From<&'a [u8; N]> for &'a ByteStr {
//     #[inline]
//     fn from(s: &'a [u8; N]) -> Self {
//         ByteStr::from_bytes(s)
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a [u8]> for &'a ByteStr {
//     #[inline]
//     fn from(s: &'a [u8]) -> Self {
//         ByteStr::from_bytes(s)
//     }
// }

// Omitted due to slice-from-array-issue-113238:
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a ByteStr> for &'a [u8] {
//     #[inline]
//     fn from(s: &'a ByteStr) -> Self {
//         &s.0
//     }
// }
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a mut ByteStr> for &'a mut [u8] {
//     #[inline]
//     fn from(s: &'a mut ByteStr) -> Self {
//         &mut s.0
//     }
// }

// Omitted due to inference failures
//
// #[unstable(feature = "bstr", issue = "134915")]
// impl<'a> From<&'a str> for &'a ByteStr {
//     #[inline]
//     fn from(s: &'a str) -> Self {
//         ByteStr::from_bytes(s.as_bytes())
//     }
// }

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<'a> const TryFrom<&'a ByteStr> for &'a str {
    type Error = crate::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a ByteStr) -> Result<Self, Self::Error> {
        crate::str::from_utf8(&s.0)
    }
}

#[unstable(feature = "bstr", issue = "134915")]
#[rustc_const_unstable(feature = "const_convert", issue = "143773")]
impl<'a> const TryFrom<&'a mut ByteStr> for &'a mut str {
    type Error = crate::str::Utf8Error;

    #[inline]
    fn try_from(s: &'a mut ByteStr) -> Result<Self, Self::Error> {
        crate::str::from_utf8_mut(&mut s.0)
    }
}
