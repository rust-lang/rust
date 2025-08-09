//! Operations on ASCII `[u8]`.

use core::ascii::EscapeDefault;

use crate::fmt::{self, Write};
use crate::intrinsics::const_eval_select;
use crate::{ascii, iter, ops};

impl [u8] {
    /// Checks if all bytes in this slice are within the ASCII range.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_slice_is_ascii", since = "1.74.0")]
    #[must_use]
    #[inline]
    pub const fn is_ascii(&self) -> bool {
        is_ascii(self)
    }

    /// If this slice [`is_ascii`](Self::is_ascii), returns it as a slice of
    /// [ASCII characters](`ascii::Char`), otherwise returns `None`.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[must_use]
    #[inline]
    pub const fn as_ascii(&self) -> Option<&[ascii::Char]> {
        if self.is_ascii() {
            // SAFETY: Just checked that it's ASCII
            Some(unsafe { self.as_ascii_unchecked() })
        } else {
            None
        }
    }

    /// Converts this slice of bytes into a slice of ASCII characters,
    /// without checking whether they're valid.
    ///
    /// # Safety
    ///
    /// Every byte in the slice must be in `0..=127`, or else this is UB.
    #[unstable(feature = "ascii_char", issue = "110998")]
    #[must_use]
    #[inline]
    pub const unsafe fn as_ascii_unchecked(&self) -> &[ascii::Char] {
        let byte_ptr: *const [u8] = self;
        let ascii_ptr = byte_ptr as *const [ascii::Char];
        // SAFETY: The caller promised all the bytes are ASCII
        unsafe { &*ascii_ptr }
    }

    /// Checks that two slices are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_eq_ignore_ascii_case", since = "1.89.0")]
    #[must_use]
    #[inline]
    pub const fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        if self.len() != other.len() {
            return false;
        }

        // FIXME(const-hack): This implementation can be reverted when
        // `core::iter::zip` is allowed in const. The original implementation:
        //  self.len() == other.len() && iter::zip(self, other).all(|(a, b)| a.eq_ignore_ascii_case(b))
        let mut a = self;
        let mut b = other;

        while let ([first_a, rest_a @ ..], [first_b, rest_b @ ..]) = (a, b) {
            if first_a.eq_ignore_ascii_case(&first_b) {
                a = rest_a;
                b = rest_b;
            } else {
                return false;
            }
        }

        true
    }

    /// Converts this slice to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_make_ascii", since = "1.84.0")]
    #[inline]
    pub const fn make_ascii_uppercase(&mut self) {
        // FIXME(const-hack): We would like to simply iterate using `for` loops but this isn't currently allowed in constant expressions.
        let mut i = 0;
        while i < self.len() {
            let byte = &mut self[i];
            byte.make_ascii_uppercase();
            i += 1;
        }
    }

    /// Converts this slice to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_stable(feature = "const_make_ascii", since = "1.84.0")]
    #[inline]
    pub const fn make_ascii_lowercase(&mut self) {
        // FIXME(const-hack): We would like to simply iterate using `for` loops but this isn't currently allowed in constant expressions.
        let mut i = 0;
        while i < self.len() {
            let byte = &mut self[i];
            byte.make_ascii_lowercase();
            i += 1;
        }
    }

    /// Returns an iterator that produces an escaped version of this slice,
    /// treating it as an ASCII string.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = b"0\t\r\n'\"\\\x9d";
    /// let escaped = s.escape_ascii().to_string();
    /// assert_eq!(escaped, "0\\t\\r\\n\\'\\\"\\\\\\x9d");
    /// ```
    #[must_use = "this returns the escaped bytes as an iterator, \
                  without modifying the original"]
    #[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
    pub fn escape_ascii(&self) -> EscapeAscii<'_> {
        EscapeAscii { inner: self.iter().flat_map(EscapeByte) }
    }

    /// Returns a byte slice with leading ASCII whitespace bytes removed.
    ///
    /// 'Whitespace' refers to the definition used by
    /// [`u8::is_ascii_whitespace`].
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(b" \t hello world\n".trim_ascii_start(), b"hello world\n");
    /// assert_eq!(b"  ".trim_ascii_start(), b"");
    /// assert_eq!(b"".trim_ascii_start(), b"");
    /// ```
    #[stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[rustc_const_stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[inline]
    pub const fn trim_ascii_start(&self) -> &[u8] {
        let mut bytes = self;
        // Note: A pattern matching based approach (instead of indexing) allows
        // making the function const.
        while let [first, rest @ ..] = bytes {
            if first.is_ascii_whitespace() {
                bytes = rest;
            } else {
                break;
            }
        }
        bytes
    }

    /// Returns a byte slice with trailing ASCII whitespace bytes removed.
    ///
    /// 'Whitespace' refers to the definition used by
    /// [`u8::is_ascii_whitespace`].
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(b"\r hello world\n ".trim_ascii_end(), b"\r hello world");
    /// assert_eq!(b"  ".trim_ascii_end(), b"");
    /// assert_eq!(b"".trim_ascii_end(), b"");
    /// ```
    #[stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[rustc_const_stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[inline]
    pub const fn trim_ascii_end(&self) -> &[u8] {
        let mut bytes = self;
        // Note: A pattern matching based approach (instead of indexing) allows
        // making the function const.
        while let [rest @ .., last] = bytes {
            if last.is_ascii_whitespace() {
                bytes = rest;
            } else {
                break;
            }
        }
        bytes
    }

    /// Returns a byte slice with leading and trailing ASCII whitespace bytes
    /// removed.
    ///
    /// 'Whitespace' refers to the definition used by
    /// [`u8::is_ascii_whitespace`].
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(b"\r hello world\n ".trim_ascii(), b"hello world");
    /// assert_eq!(b"  ".trim_ascii(), b"");
    /// assert_eq!(b"".trim_ascii(), b"");
    /// ```
    #[stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[rustc_const_stable(feature = "byte_slice_trim_ascii", since = "1.80.0")]
    #[inline]
    pub const fn trim_ascii(&self) -> &[u8] {
        self.trim_ascii_start().trim_ascii_end()
    }
}

impl_fn_for_zst! {
    #[derive(Clone)]
    struct EscapeByte impl Fn = |byte: &u8| -> ascii::EscapeDefault {
        ascii::escape_default(*byte)
    };
}

/// An iterator over the escaped version of a byte slice.
///
/// This `struct` is created by the [`slice::escape_ascii`] method. See its
/// documentation for more information.
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct EscapeAscii<'a> {
    inner: iter::FlatMap<super::Iter<'a, u8>, ascii::EscapeDefault, EscapeByte>,
}

#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
impl<'a> iter::Iterator for EscapeAscii<'a> {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<u8> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
    #[inline]
    fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R
    where
        Fold: FnMut(Acc, Self::Item) -> R,
        R: ops::Try<Output = Acc>,
    {
        self.inner.try_fold(init, fold)
    }
    #[inline]
    fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
    where
        Fold: FnMut(Acc, Self::Item) -> Acc,
    {
        self.inner.fold(init, fold)
    }
    #[inline]
    fn last(mut self) -> Option<u8> {
        self.next_back()
    }
}

#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
impl<'a> iter::DoubleEndedIterator for EscapeAscii<'a> {
    fn next_back(&mut self) -> Option<u8> {
        self.inner.next_back()
    }
}
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
impl<'a> iter::FusedIterator for EscapeAscii<'a> {}
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
impl<'a> fmt::Display for EscapeAscii<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // disassemble iterator, including front/back parts of flatmap in case it has been partially consumed
        let (front, slice, back) = self.clone().inner.into_parts();
        let front = front.unwrap_or(EscapeDefault::empty());
        let mut bytes = slice.unwrap_or_default().as_slice();
        let back = back.unwrap_or(EscapeDefault::empty());

        // usually empty, so the formatter won't have to do any work
        for byte in front {
            f.write_char(byte as char)?;
        }

        fn needs_escape(b: u8) -> bool {
            b > 0x7E || b < 0x20 || b == b'\\' || b == b'\'' || b == b'"'
        }

        while bytes.len() > 0 {
            // fast path for the printable, non-escaped subset of ascii
            let prefix = bytes.iter().take_while(|&&b| !needs_escape(b)).count();
            // SAFETY: prefix length was derived by counting bytes in the same splice, so it's in-bounds
            let (prefix, remainder) = unsafe { bytes.split_at_unchecked(prefix) };
            // SAFETY: prefix is a valid utf8 sequence, as it's a subset of ASCII
            let prefix = unsafe { crate::str::from_utf8_unchecked(prefix) };

            f.write_str(prefix)?; // the fast part

            bytes = remainder;

            if let Some(&b) = bytes.first() {
                // guaranteed to be non-empty, better to write it as a str
                fmt::Display::fmt(&ascii::escape_default(b), f)?;
                bytes = &bytes[1..];
            }
        }

        // also usually empty
        for byte in back {
            f.write_char(byte as char)?;
        }
        Ok(())
    }
}
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
impl<'a> fmt::Debug for EscapeAscii<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EscapeAscii").finish_non_exhaustive()
    }
}

#[inline]
#[rustc_allow_const_fn_unstable(const_eval_select)] // fallback impl has same behavior
const fn is_ascii(bytes: &[u8]) -> bool {
    // The runtime version behaves the same as the compiletime version, it's
    // just more optimized.
    const_eval_select!(
        @capture { bytes: &[u8] } -> bool:
        if const {
            is_ascii_const(bytes)
        } else {
            if cfg!(all(target_arch = "x86_64", target_feature = "sse2")) {
                is_ascii_simd::<32>(bytes)
            } else if cfg!(target_arch = "aarch64") {
                is_ascii_swar::<4>(bytes)
            } else {
                is_ascii_swar::<2>(bytes)
            }
        }
    )
}

#[inline]
const fn is_ascii_const(mut bytes: &[u8]) -> bool {
    while let [first, rest @ ..] = bytes {
        if !first.is_ascii() {
            break;
        }
        bytes = rest;
    }
    bytes.is_empty()
}

#[inline(always)]
fn is_ascii_scalar(bytes: &[u8]) -> bool {
    bytes.iter().all(u8::is_ascii)
}

#[inline(always)]
fn is_ascii_word(word: usize) -> bool {
    word & usize::repeat_u8(0x80) == 0
}

/// Check `bytes` are ASCII by reading `UNROLL_FACTOR` words at a time.
#[inline(always)]
#[unstable(feature = "str_internals", issue = "none")]
pub fn is_ascii_swar<const UNROLL_FACTOR: usize>(bytes: &[u8]) -> bool {
    if bytes.len() < size_of::<usize>() {
        return is_ascii_scalar(bytes);
    }

    // SAFETY: Casting between `u8` and `usize` is fine.
    let (_, words, _) = unsafe { bytes.align_to::<usize>() };
    let crate::ops::Range { start, end } = bytes.as_ptr_range();

    // SAFETY: checked above that `len >= size_of::<usize>()`.
    let first_word = unsafe { start.cast::<usize>().read_unaligned() };
    if !is_ascii_word(first_word) {
        return false;
    }

    let (chunks, remainder) = words.as_chunks::<UNROLL_FACTOR>();
    for chunk in chunks {
        let word = chunk.iter().fold(0, |acc, word| word | acc);
        if !is_ascii_word(word) {
            return false;
        }
    }

    for word in remainder {
        if !is_ascii_word(*word) {
            return false;
        }
    }

    // SAFETY: checked above that `len >= size_of::<usize>()`.
    let last_word = unsafe { end.cast::<usize>().sub(1).read_unaligned() };
    if !is_ascii_word(last_word) {
        return false;
    }

    true
}

/// Check `bytes` are ASCII by reading `CHUNK_SIZE` bytes at a time.
#[inline(always)]
#[unstable(feature = "str_internals", issue = "none")]
pub fn is_ascii_simd<const CHUNK_SIZE: usize>(bytes: &[u8]) -> bool {
    let (chunks, remainder) = bytes.as_chunks::<CHUNK_SIZE>();
    chunks.iter().all(|chunk| is_ascii_scalar(chunk)) && is_ascii_scalar(remainder)
}
