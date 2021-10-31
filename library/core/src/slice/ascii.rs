//! Operations on ASCII `[u8]`.

use crate::ascii;
use crate::fmt::{self, Write};
use crate::iter;
use crate::mem;
use crate::ops;

#[lang = "slice_u8"]
#[cfg(not(test))]
impl [u8] {
    /// Checks if all bytes in this slice are within the ASCII range.
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[must_use]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        is_ascii(self)
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
    /// #![feature(inherent_ascii_escape)]
    ///
    /// let s = b"0\t\r\n'\"\\\x9d";
    /// let escaped = s.escape_ascii().to_string();
    /// assert_eq!(escaped, "0\\t\\r\\n\\'\\\"\\\\\\x9d");
    /// ```
    #[must_use = "this returns the escaped bytes as an iterator, \
                  without modifying the original"]
    #[unstable(feature = "inherent_ascii_escape", issue = "77174")]
    pub fn escape_ascii(&self) -> EscapeAscii<'_> {
        EscapeAscii { inner: self.iter().flat_map(EscapeByte) }
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
#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
#[derive(Clone)]
pub struct EscapeAscii<'a> {
    inner: iter::FlatMap<super::Iter<'a, u8>, ascii::EscapeDefault, EscapeByte>,
}

#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
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

#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
impl<'a> iter::DoubleEndedIterator for EscapeAscii<'a> {
    fn next_back(&mut self) -> Option<u8> {
        self.inner.next_back()
    }
}
#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
impl<'a> iter::ExactSizeIterator for EscapeAscii<'a> {}
#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
impl<'a> iter::FusedIterator for EscapeAscii<'a> {}
#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
impl<'a> fmt::Display for EscapeAscii<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.clone().try_for_each(|b| f.write_char(b as char))
    }
}
#[unstable(feature = "inherent_ascii_escape", issue = "77174")]
impl<'a> fmt::Debug for EscapeAscii<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EscapeAscii").finish_non_exhaustive()
    }
}

/// Returns `true` if any byte in the word `v` is nonascii (>= 128). Snarfed
/// from `../str/mod.rs`, which does something similar for utf8 validation.
#[inline]
fn contains_nonascii(v: usize) -> bool {
    const NONASCII_MASK: usize = 0x80808080_80808080u64 as usize;
    (NONASCII_MASK & v) != 0
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
fn is_ascii(s: &[u8]) -> bool {
    const USIZE_SIZE: usize = mem::size_of::<usize>();

    let len = s.len();
    let align_offset = s.as_ptr().align_offset(USIZE_SIZE);

    // If we wouldn't gain anything from the word-at-a-time implementation, fall
    // back to a scalar loop.
    //
    // We also do this for architectures where `size_of::<usize>()` isn't
    // sufficient alignment for `usize`, because it's a weird edge case.
    if len < USIZE_SIZE || len < align_offset || USIZE_SIZE < mem::align_of::<usize>() {
        return s.iter().all(|b| b.is_ascii());
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
    debug_assert_eq!((word_ptr as usize) % mem::align_of::<usize>(), 0);

    // Read subsequent words until the last aligned word, excluding the last
    // aligned word by itself to be done in tail check later, to ensure that
    // tail is always one `usize` at most to extra branch `byte_pos == len`.
    while byte_pos < len - USIZE_SIZE {
        debug_assert!(
            // Sanity check that the read is in bounds
            (word_ptr as usize + USIZE_SIZE) <= (start.wrapping_add(len) as usize) &&
            // And that our assumptions about `byte_pos` hold.
            (word_ptr as usize) - (start as usize) == byte_pos
        );

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
