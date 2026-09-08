//! Operations on ASCII `[u8]`.

use core::ascii::EscapeDefault;

use crate::fmt::{self, Write};
#[cfg(not(all(target_arch = "loongarch64", target_feature = "lsx")))]
use crate::intrinsics::const_eval_select;
use crate::{ascii, iter, ops};
#[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
use crate::{
    iter::{Filter, FusedIterator},
    slice::Split,
    str::{BytesIsNotEmpty, IsAsciiWhitespace},
};

impl [u8] {
    /// Checks if all bytes in this slice are within the ASCII range.
    ///
    /// An empty slice returns `true`.
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

        #[cfg(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            all(target_arch = "aarch64", target_feature = "neon")
        ))]
        {
            const CHUNK_SIZE: usize = 16;
            // The following function has two invariants:
            // 1. The slice lengths must be equal, which we checked above.
            // 2. The slice lengths must greater than or equal to N, which this
            //    if-statement is checking.
            if self.len() >= CHUNK_SIZE {
                return self.eq_ignore_ascii_case_chunks::<CHUNK_SIZE>(other);
            }
        }

        self.eq_ignore_ascii_case_simple(other)
    }

    /// ASCII case-insensitive equality check without chunk-at-a-time
    /// optimization.
    #[inline]
    const fn eq_ignore_ascii_case_simple(&self, other: &[u8]) -> bool {
        // FIXME(const-hack): This implementation can be reverted when
        // `core::iter::zip` is allowed in const. The original implementation:
        //  self.len() == other.len() && iter::zip(self, other).all(|(a, b)| a.eq_ignore_ascii_case(b))
        let mut a = self;
        let mut b = other;

        while let ([first_a, rest_a @ ..], [first_b, rest_b @ ..]) = (a, b) {
            if first_a.eq_ignore_ascii_case(first_b) {
                a = rest_a;
                b = rest_b;
            } else {
                return false;
            }
        }

        true
    }

    /// Optimized version of `eq_ignore_ascii_case` to process chunks at a time.
    ///
    /// Platforms that have SIMD instructions may benefit from this
    /// implementation over `eq_ignore_ascii_case_simple`.
    ///
    /// # Invariants
    ///
    /// The caller must guarantee that the slices are equal in length, and the
    /// slice lengths are greater than or equal to `N` bytes.
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    #[inline]
    const fn eq_ignore_ascii_case_chunks<const N: usize>(&self, other: &[u8]) -> bool {
        // FIXME(const-hack): The while-loops that follow should be replaced by
        // for-loops when available in const.

        let (self_chunks, self_rem) = self.as_chunks::<N>();
        let (other_chunks, _) = other.as_chunks::<N>();

        // Branchless check to encourage auto-vectorization
        #[inline(always)]
        const fn eq_ignore_ascii_inner<const L: usize>(lhs: &[u8; L], rhs: &[u8; L]) -> bool {
            let mut equal_ascii = true;
            let mut j = 0;
            while j < L {
                equal_ascii &= lhs[j].eq_ignore_ascii_case(&rhs[j]);
                j += 1;
            }

            equal_ascii
        }

        // Process the chunks, returning early if an inequality is found
        let mut i = 0;
        while i < self_chunks.len() && i < other_chunks.len() {
            if !eq_ignore_ascii_inner(&self_chunks[i], &other_chunks[i]) {
                return false;
            }
            i += 1;
        }

        // Check the length invariant which is necessary for the tail-handling
        // logic to be correct. This should have been upheld by the caller,
        // otherwise lengths less than N will compare as true without any
        // checking.
        debug_assert!(self.len() >= N);

        // If there are remaining tails, load the last N bytes in the slices to
        // avoid falling back to per-byte checking.
        if !self_rem.is_empty() {
            if let (Some(a_rem), Some(b_rem)) = (self.last_chunk::<N>(), other.last_chunk::<N>()) {
                if !eq_ignore_ascii_inner(a_rem, b_rem) {
                    return false;
                }
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
    /// [`u8::is_ascii_whitespace`]. Importantly, this definition excludes
    /// the `\0x0B` byte even though it has the Unicode [`White_Space`] property
    /// and is removed by [`str::trim_start`].
    ///
    /// [`White_Space`]: https://www.unicode.org/reports/tr44/#White_Space
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
    /// [`u8::is_ascii_whitespace`]. Importantly, this definition excludes
    /// the `\0x0B` byte even though it has the Unicode [`White_Space`] property
    /// and is removed by [`str::trim_end`].
    ///
    /// [`White_Space`]: https://www.unicode.org/reports/tr44/#White_Space
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
    /// [`u8::is_ascii_whitespace`]. Importantly, this definition excludes
    /// the `\0x0B` byte even though it has the Unicode [`White_Space`] property
    /// and is removed by [`str::trim`].
    ///
    /// [`White_Space`]: https://www.unicode.org/reports/tr44/#White_Space
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

    /// Splits a byte slice by ASCII whitespace.
    ///
    /// The returned iterator yields byte slices that are subslices of the
    /// original byte slice, separated by any amount of ASCII whitespace.
    ///
    /// This uses the same definition as [`u8::is_ascii_whitespace`].
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(u8_split_ascii_whitespace)]
    ///
    /// let mut iter = b"A few words".split_ascii_whitespace();
    ///
    /// assert_eq!(Some(&b"A"[..]), iter.next());
    /// assert_eq!(Some(&b"few"[..]), iter.next());
    /// assert_eq!(Some(&b"words"[..]), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// Various kinds of ASCII whitespace are considered
    /// (see [`u8::is_ascii_whitespace`]):
    ///
    /// ```
    /// #![feature(u8_split_ascii_whitespace)]
    ///
    /// let mut iter = b" Mary   had\ta little  \n\t lamb".split_ascii_whitespace();
    ///
    /// assert_eq!(Some(&b"Mary"[..]), iter.next());
    /// assert_eq!(Some(&b"had"[..]), iter.next());
    /// assert_eq!(Some(&b"a"[..]), iter.next());
    /// assert_eq!(Some(&b"little"[..]), iter.next());
    /// assert_eq!(Some(&b"lamb"[..]), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// If the byte slice is empty or contains only ASCII whitespace, the iterator
    /// yields no byte slices:
    ///
    /// ```
    /// #![feature(u8_split_ascii_whitespace)]
    ///
    /// assert_eq!(b"".split_ascii_whitespace().next(), None);
    /// assert_eq!(b"   ".split_ascii_whitespace().next(), None);
    /// ```
    #[must_use = "this returns the split byte slice as an iterator, without modifying the original"]
    #[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
    #[inline]
    pub fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace<'_> {
        let inner = self.split(IsAsciiWhitespace).filter(BytesIsNotEmpty);
        SplitAsciiWhitespace { inner }
    }
}

/// An iterator over the non-ASCII-whitespace subslices of a byte slice,
/// separated by any amount of ASCII whitespace.
///
/// This struct is created by the [`split_ascii_whitespace`] method on [`[u8]`][byteslice].
/// See its documentation for more.
///
/// [`split_ascii_whitespace`]: slice::split_ascii_whitespace
/// [byteslice]: prim@slice
#[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
#[derive(Clone, Debug)]
pub struct SplitAsciiWhitespace<'a> {
    pub(crate) inner: Filter<Split<'a, u8, IsAsciiWhitespace>, BytesIsNotEmpty>,
}

#[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
impl<'a> Iterator for SplitAsciiWhitespace<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<&'a [u8]> {
        self.next_back()
    }
}

#[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
impl<'a> DoubleEndedIterator for SplitAsciiWhitespace<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a [u8]> {
        self.inner.next_back()
    }
}

#[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
impl FusedIterator for SplitAsciiWhitespace<'_> {}

impl<'a> SplitAsciiWhitespace<'a> {
    /// Returns remainder of the split slice.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(u8_split_ascii_whitespace)]
    ///
    /// let mut split = b"Mary had a little lamb".split_ascii_whitespace();
    /// assert_eq!(split.remainder(), Some(b"Mary had a little lamb".as_slice()));
    ///
    /// split.next();
    /// assert_eq!(split.remainder(), Some(b"had a little lamb".as_slice()));
    ///
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[must_use]
    // This is also blocked on: https://github.com/rust-lang/rust/issues/77998
    #[unstable(feature = "u8_split_ascii_whitespace", issue = "147878")]
    pub fn remainder(&self) -> Option<&'a [u8]> {
        if self.inner.iter.finished {
            return None;
        }

        Some(self.inner.iter.v)
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

        while !bytes.is_empty() {
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
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "loongarch64", target_feature = "lsx"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
#[inline]
#[rustc_allow_const_fn_unstable(const_eval_select)] // fallback impl has same behavior
const fn is_ascii(s: &[u8]) -> bool {
    // The runtime version behaves the same as the compiletime version, it's
    // just more optimized.
    const_eval_select!(
        @capture { s: &[u8] } -> bool:
        if const {
            is_ascii_simple(s)
        } else {
            /// Returns `true` if any byte in the word `v` is nonascii (>= 128). Snarfed
            /// from `../str/mod.rs`, which does something similar for utf8 validation.
            const fn contains_nonascii(v: usize) -> bool {
                const NONASCII_MASK: usize = usize::repeat_u8(0x80);
                (NONASCII_MASK & v) != 0
            }

            const USIZE_SIZE: usize = size_of::<usize>();

            let len = s.len();
            let align_offset = s.as_ptr().align_offset(USIZE_SIZE);

            // If we wouldn't gain anything from the word-at-a-time implementation, fall
            // back to a scalar loop.
            //
            // We also do this for architectures where `size_of::<usize>()` isn't
            // sufficient alignment for `usize`, because it's a weird edge case.
            if len < USIZE_SIZE || len < align_offset || USIZE_SIZE < align_of::<usize>() {
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
            debug_assert!(word_ptr.is_aligned_to(align_of::<usize>()));

            // Read subsequent words until the last aligned word, excluding the last
            // aligned word by itself to be done in tail check later, to ensure that
            // tail is always one `usize` at most to extra branch `byte_pos == len`.
            while byte_pos < len - USIZE_SIZE {
                // Sanity check that the read is in bounds
                debug_assert!(byte_pos + USIZE_SIZE <= len);
                // And that our assumptions about `byte_pos` hold.
                debug_assert!(word_ptr.cast::<u8>() == start.wrapping_add(byte_pos));

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
    )
}

/// Chunk size for SSE2 vectorized ASCII checking (4x 16-byte loads).
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
const SSE2_CHUNK_SIZE: usize = 64;

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn is_ascii_sse2(bytes: &[u8]) -> bool {
    use crate::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128};

    let (chunks, rest) = bytes.as_chunks::<SSE2_CHUNK_SIZE>();

    for chunk in chunks {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is 64 bytes. SSE2 is baseline on x86_64.
        let mask = unsafe {
            let a1 = _mm_loadu_si128(ptr as *const __m128i);
            let a2 = _mm_loadu_si128(ptr.add(16) as *const __m128i);
            let b1 = _mm_loadu_si128(ptr.add(32) as *const __m128i);
            let b2 = _mm_loadu_si128(ptr.add(48) as *const __m128i);
            // OR all chunks - if any byte has high bit set, combined will too.
            let combined = _mm_or_si128(_mm_or_si128(a1, a2), _mm_or_si128(b1, b2));
            // Create a mask from the MSBs of each byte.
            // If any byte is >= 128, its MSB is 1, so the mask will be non-zero.
            _mm_movemask_epi8(combined)
        };
        if mask != 0 {
            return false;
        }
    }

    // Handle remaining bytes
    rest.iter().all(|b| b.is_ascii())
}

/// Chunk size for NEON vectorized ASCII checking (4x 16-byte loads).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const NEON_CHUNK_SIZE: usize = 64;

/// Width of a single NEON vector, used to vectorize the tail left over by the
/// unrolled `NEON_CHUNK_SIZE` loop.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
const NEON_VECTOR_SIZE: usize = 16;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
fn is_ascii_neon(bytes: &[u8]) -> bool {
    use crate::arch::aarch64::{vld1q_u8, vmaxvq_u8, vorrq_u8};

    let (chunks, rest) = bytes.as_chunks::<NEON_CHUNK_SIZE>();

    for chunk in chunks {
        let ptr = chunk.as_ptr();
        // SAFETY: chunk is 64 bytes, and `vld1q_u8` has no alignment requirement.
        let max = unsafe {
            let a1 = vld1q_u8(ptr);
            let a2 = vld1q_u8(ptr.add(16));
            let b1 = vld1q_u8(ptr.add(32));
            let b2 = vld1q_u8(ptr.add(48));
            // OR all chunks - if any byte has high bit set, combined will too.
            let combined = vorrq_u8(vorrq_u8(a1, a2), vorrq_u8(b1, b2));
            // `vmaxvq_u8` is a horizontal reduction with a longer latency than
            // `vorrq_u8`, so it runs once per 64 bytes rather than once per load.
            vmaxvq_u8(combined)
        };
        if max >= 128 {
            return false;
        }
    }

    // The unrolled loop above leaves up to 63 bytes, so sweep those a vector at
    // a time before falling back to a byte-at-a-time check.
    let (vectors, rest) = rest.as_chunks::<NEON_VECTOR_SIZE>();

    for vector in vectors {
        // SAFETY: vector is 16 bytes, and `vld1q_u8` has no alignment requirement.
        let max = unsafe { vmaxvq_u8(vld1q_u8(vector.as_ptr())) };
        if max >= 128 {
            return false;
        }
    }

    // Handle remaining bytes
    rest.iter().all(|b| b.is_ascii())
}

/// Uses explicit SIMD intrinsics to prevent LLVM from auto-vectorizing with
/// broken code (e.g., AVX-512 on x86-64 that extracts mask bits one-by-one).
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(target_arch = "aarch64", target_feature = "neon")
))]
#[inline]
#[rustc_allow_const_fn_unstable(const_eval_select)]
const fn is_ascii(bytes: &[u8]) -> bool {
    const USIZE_SIZE: usize = size_of::<usize>();
    const NONASCII_MASK: usize = usize::MAX / 255 * 0x80;

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    const SIMD_MIN_LEN: usize = SSE2_CHUNK_SIZE;
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    const SIMD_MIN_LEN: usize = NEON_CHUNK_SIZE;

    const_eval_select!(
        @capture { bytes: &[u8] } -> bool:
        if const {
            is_ascii_simple(bytes)
        } else {
            // For small inputs, use usize-at-a-time processing to avoid SSE2 call overhead.
            if bytes.len() < SIMD_MIN_LEN {
                let (chunks, remainder) = bytes.as_chunks::<USIZE_SIZE>();
                for chunk in chunks {
                    let word = usize::from_ne_bytes(*chunk);
                    if (word & NONASCII_MASK) != 0 {
                        return false;
                    }
                }
                return remainder.iter().all(|b| b.is_ascii());
            }

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            { is_ascii_sse2(bytes) }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            { is_ascii_neon(bytes) }
        }
    )
}

/// ASCII test optimized to use the `vmskltz.b` instruction on `loongarch64`.
///
/// Other platforms are not likely to benefit from this code structure, so they
/// use SWAR techniques to test for ASCII in `usize`-sized chunks.
#[cfg(all(target_arch = "loongarch64", target_feature = "lsx"))]
#[inline]
const fn is_ascii(bytes: &[u8]) -> bool {
    // Process chunks of 32 bytes at a time in the fast path to enable
    // auto-vectorization and use of `vmskltz.b`. Two 128-bit vector registers
    // can be OR'd together and then the resulting vector can be tested for
    // non-ASCII bytes.
    const CHUNK_SIZE: usize = 32;

    let mut i = 0;

    while i + CHUNK_SIZE <= bytes.len() {
        let chunk_end = i + CHUNK_SIZE;

        // Get LLVM to produce a `vmskltz.b` instruction on loongarch64 which
        // creates a mask from the most significant bit of each byte.
        // ASCII bytes are less than 128 (0x80), so their most significant
        // bit is unset.
        let mut count = 0;
        while i < chunk_end {
            count += bytes[i].is_ascii() as u8;
            i += 1;
        }

        // All bytes should be <= 127 so count is equal to chunk size.
        if count != CHUNK_SIZE as u8 {
            return false;
        }
    }

    // Process the remaining `bytes.len() % N` bytes.
    let mut is_ascii = true;
    while i < bytes.len() {
        is_ascii &= bytes[i].is_ascii();
        i += 1;
    }

    is_ascii
}
