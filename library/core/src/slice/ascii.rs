//! Operations on ASCII `[u8]`.

use crate::ascii;
use crate::fmt::{self, Write};
use crate::iter;
use crate::mem;
use crate::ops;

#[cfg(not(test))]
impl [u8] {
    /// Checks if all bytes in this slice are within the ASCII range.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[rustc_const_unstable(feature = "const_slice_is_ascii", issue = "111090")]
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
    #[must_use]
    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &[u8]) -> bool {
        self.len() == other.len() && iter::zip(self, other).all(|(a, b)| a.eq_ignore_ascii_case(b))
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
    #[inline]
    pub fn make_ascii_uppercase(&mut self) {
        for byte in self {
            byte.make_ascii_uppercase();
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
    #[inline]
    pub fn make_ascii_lowercase(&mut self) {
        for byte in self {
            byte.make_ascii_lowercase();
        }
    }

    /// Returns an iterator that produces an escaped version of this slice,
    /// treating it as an ASCII string.
    ///
    /// # Examples
    ///
    /// ```
    ///
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
    /// `u8::is_ascii_whitespace`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(byte_slice_trim_ascii)]
    ///
    /// assert_eq!(b" \t hello world\n".trim_ascii_start(), b"hello world\n");
    /// assert_eq!(b"  ".trim_ascii_start(), b"");
    /// assert_eq!(b"".trim_ascii_start(), b"");
    /// ```
    #[unstable(feature = "byte_slice_trim_ascii", issue = "94035")]
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
    /// `u8::is_ascii_whitespace`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(byte_slice_trim_ascii)]
    ///
    /// assert_eq!(b"\r hello world\n ".trim_ascii_end(), b"\r hello world");
    /// assert_eq!(b"  ".trim_ascii_end(), b"");
    /// assert_eq!(b"".trim_ascii_end(), b"");
    /// ```
    #[unstable(feature = "byte_slice_trim_ascii", issue = "94035")]
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
    /// `u8::is_ascii_whitespace`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(byte_slice_trim_ascii)]
    ///
    /// assert_eq!(b"\r hello world\n ".trim_ascii(), b"hello world");
    /// assert_eq!(b"  ".trim_ascii(), b"");
    /// assert_eq!(b"".trim_ascii(), b"");
    /// ```
    #[unstable(feature = "byte_slice_trim_ascii", issue = "94035")]
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
        self.clone().try_for_each(|b| f.write_char(b as char))
    }
}
#[stable(feature = "inherent_ascii_escape", since = "1.60.0")]
impl<'a> fmt::Debug for EscapeAscii<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EscapeAscii").finish_non_exhaustive()
    }
}

/// Returns `true` if any byte in the word `v` is nonascii (>= 128). Snarfed
/// from `../str/mod.rs`, which does something similar for utf8 validation.
#[inline]
const fn contains_nonascii(v: usize) -> bool {
    const NONASCII_MASK: usize = usize::repeat_u8(0x80);
    (NONASCII_MASK & v) != 0
}

/// ASCII test *without* the chunk-at-a-time optimizations.
///
/// This is carefully structured to produce nice small code -- it's smaller in
/// `-O` than what the "obvious" ways produces under `-C opt-level=s`.  If you
/// touch it, be sure to run (and update if needed) the assembly test.
#[unstable(feature = "str_internals", issue = "none")]
#[doc(hidden)]
#[inline]
pub const fn is_ascii_simple(mut bytes: &[u8]) -> bool {
    while let [rest @ .., last] = bytes {
        if !last.is_ascii() {
            break;
        }
        bytes = rest;
    }
    bytes.is_empty()
}

/// Optimized ASCII test that will use usize-at-a-time operations instead of
/// byte-at-a-time operations (when possible).
///
/// The algorithm we use here is pretty simple. If `s` is too short, we just
/// check each byte and be done with it. Otherwise:
///
/// - Read the first word with an unaligned load.
/// - Align the pointer, read subsequent words until end with aligned loads.
/// - Read the last `usize` from `s` with an unaligned load.
///
/// If any of these loads produces something for which `contains_nonascii`
/// (above) returns true, then we know the answer is false.
#[inline]
const fn is_ascii(s: &[u8]) -> bool {
    const USIZE_SIZE: usize = mem::size_of::<usize>();

    let len = s.len();
    let align_offset = s.as_ptr().align_offset(USIZE_SIZE);

    // If we wouldn't gain anything from the word-at-a-time implementation, fall
    // back to a scalar loop.
    //
    // We also do this for architectures where `size_of::<usize>()` isn't
    // sufficient alignment for `usize`, because it's a weird edge case.
    if len < USIZE_SIZE || len < align_offset || USIZE_SIZE < mem::align_of::<usize>() {
        return is_ascii_simple(s);
    }

    // We always read the first word unaligned, which means `align_offset` is
    // 0, we'd read the same value again for the aligned read.
    let offset_to_aligned = if align_offset == 0 { USIZE_SIZE } else { align_offset };

    let start = s.as_ptr();
    // SAFETY: We verify `len < USIZE_SIZE` above.
    let first_word = unsafe { (start as *const usize).read_unaligned() };

    if contains_nonascii(first_word) {
        return false;
    }
    // We checked this above, somewhat implicitly. Note that `offset_to_aligned`
    // is either `align_offset` or `USIZE_SIZE`, both of are explicitly checked
    // above.
    debug_assert!(offset_to_aligned <= len);

    // SAFETY: word_ptr is the (properly aligned) usize ptr we use to read the
    // middle chunk of the slice.
    let mut word_ptr = unsafe { start.add(offset_to_aligned) as *const usize };

    // `byte_pos` is the byte index of `word_ptr`, used for loop end checks.
    let mut byte_pos = offset_to_aligned;

    // Paranoia check about alignment, since we're about to do a bunch of
    // unaligned loads. In practice this should be impossible barring a bug in
    // `align_offset` though.
    // While this method is allowed to spuriously fail in CTFE, if it doesn't
    // have alignment information it should have given a `usize::MAX` for
    // `align_offset` earlier, sending things through the scalar path instead of
    // this one, so this check should pass if it's reachable.
    debug_assert!(word_ptr.is_aligned_to(mem::align_of::<usize>()));

    // Read subsequent words until the last aligned word, excluding the last
    // aligned word by itself to be done in tail check later, to ensure that
    // tail is always one `usize` at most to extra branch `byte_pos == len`.
    while byte_pos < len - USIZE_SIZE {
        // Sanity check that the read is in bounds
        debug_assert!(byte_pos + USIZE_SIZE <= len);
        // And that our assumptions about `byte_pos` hold.
        debug_assert!(matches!(
            word_ptr.cast::<u8>().guaranteed_eq(start.wrapping_add(byte_pos)),
            // These are from the same allocation, so will hopefully always be
            // known to match even in CTFE, but if it refuses to compare them
            // that's ok since it's just a debug check anyway.
            None | Some(true),
        ));

        // SAFETY: We know `word_ptr` is properly aligned (because of
        // `align_offset`), and we know that we have enough bytes between `word_ptr` and the end
        let word = unsafe { word_ptr.read() };
        if contains_nonascii(word) {
            return false;
        }

        byte_pos += USIZE_SIZE;
        // SAFETY: We know that `byte_pos <= len - USIZE_SIZE`, which means that
        // after this `add`, `word_ptr` will be at most one-past-the-end.
        word_ptr = unsafe { word_ptr.add(1) };
    }

    // Sanity check to ensure there really is only one `usize` left. This should
    // be guaranteed by our loop condition.
    debug_assert!(byte_pos <= len && len - byte_pos <= USIZE_SIZE);

    // SAFETY: This relies on `len >= USIZE_SIZE`, which we check at the start.
    let last_word = unsafe { (start.add(len - USIZE_SIZE) as *const usize).read_unaligned() };

    !contains_nonascii(last_word)
}
