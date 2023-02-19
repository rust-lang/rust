//! [The Pattern API] implementation for searching in `&str`.
//!
//! The implementation provides generic mechanism for using different pattern
//! types when searching through a string.  Although this API is unstable, it is
//! exposed via stable APIs on the [`str`] type.
//!
//! Depending on the type of the pattern, the behaviour of methods like
//! [`str::find`] and [`str::contains`] can change. The table below describes
//! some of those behaviours.
//!
//! | Pattern type             | Match condition                           |
//! |--------------------------|-------------------------------------------|
//! | `&str`                   | is substring                              |
//! | `char`                   | is contained in string                    |
//! | `&[char]`                | any char in slice is contained in string  |
//! | `F: FnMut(char) -> bool` | `F` returns `true` for a char in string   |
//! | `&&str`                  | is substring                              |
//! | `&String`                | is substring                              |
//!
//! # Examples
//!
//! ```
//! let s = "Can you find a needle in a haystack?";
//!
//! // &str pattern
//! assert_eq!(s.find("you"), Some(4));
//! assert_eq!(s.find("thou"), None);
//!
//! // char pattern
//! assert_eq!(s.find('n'), Some(2));
//! assert_eq!(s.find('N'), None);
//!
//! // Array of chars pattern and slices thereof
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u']), Some(1));
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u'][..]), Some(1));
//! assert_eq!(s.find(&['q', 'v', 'x']), None);
//!
//! // Predicate closure
//! assert_eq!(s.find(|c: char| c.is_ascii_punctuation()), Some(35));
//! assert_eq!(s.find(|c: char| c.is_lowercase()), Some(1));
//! assert_eq!(s.find(|c: char| !c.is_ascii()), None);
//! ```
//!
//! [The Pattern API]: crate::pattern

#![unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]

use crate::cmp::Ordering;
use crate::fmt;
use crate::ops::Range;
use crate::pattern::{
    DoubleEndedSearcher, Haystack, Pattern, ReverseSearcher, SearchStep, Searcher,
};
use crate::str_bytes;

/////////////////////////////////////////////////////////////////////////////
// Impl for Haystack
/////////////////////////////////////////////////////////////////////////////

impl<'a> Haystack for &'a str {
    type Cursor = usize;

    #[inline(always)]
    fn cursor_at_front(self) -> usize {
        0
    }
    #[inline(always)]
    fn cursor_at_back(self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn is_empty(self) -> bool {
        self.is_empty()
    }

    #[inline(always)]
    unsafe fn get_unchecked(self, range: Range<usize>) -> Self {
        // SAFETY: Caller promises position is a character boundary.
        unsafe { self.get_unchecked(range) }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for char
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<char as Pattern<H>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharSearcher<'a>(str_bytes::CharSearcher<'a>);

impl<'a> CharSearcher<'a> {
    fn new(haystack: &'a str, chr: char) -> Self {
        Self(str_bytes::CharSearcher::new(str_bytes::Bytes::from(haystack), chr))
    }
}

unsafe impl<'a> Searcher<&'a str> for CharSearcher<'a> {
    #[inline]
    fn haystack(&self) -> &'a str {
        // SAFETY: self.0’s haystack was created from &str thus it is valid
        // UTF-8.
        unsafe { super::from_utf8_unchecked(self.0.haystack().as_bytes()) }
    }
    #[inline]
    fn next(&mut self) -> SearchStep {
        self.0.next()
    }
    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.0.next_match()
    }
    #[inline]
    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject()
    }
}

unsafe impl<'a> ReverseSearcher<&'a str> for CharSearcher<'a> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        self.0.next_back()
    }
    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_match_back()
    }
    #[inline]
    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject_back()
    }
}

impl<'a> DoubleEndedSearcher<&'a str> for CharSearcher<'a> {}

/// Searches for chars that are equal to a given [`char`].
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find('o'), Some(4));
/// assert_eq!("Hello world".find('x'), None);
/// ```
impl<'a> Pattern<&'a str> for char {
    type Searcher = CharSearcher<'a>;

    #[inline]
    fn into_searcher(self, haystack: &'a str) -> Self::Searcher {
        CharSearcher::new(haystack, self)
    }

    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        self.is_contained_in(str_bytes::Bytes::from(haystack))
    }

    #[inline]
    fn is_prefix_of(self, haystack: &'a str) -> bool {
        self.encode_utf8(&mut [0u8; 4]).is_prefix_of(haystack)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &'a str) -> Option<&'a str> {
        self.strip_prefix_of(str_bytes::Bytes::from(haystack)).map(|bytes| {
            // SAFETY: Bytes were created from &str and Bytes never splits
            // inside of UTF-8 bytes sequences thus `bytes` is still valid
            // UTF-8.
            unsafe { super::from_utf8_unchecked(bytes.as_bytes()) }
        })
    }

    #[inline]
    fn is_suffix_of(self, haystack: &'a str) -> bool {
        self.is_suffix_of(str_bytes::Bytes::from(haystack))
    }

    #[inline]
    fn strip_suffix_of(self, haystack: &'a str) -> Option<&'a str> {
        self.strip_suffix_of(str_bytes::Bytes::from(haystack)).map(|bytes| {
            // SAFETY: Bytes were created from &str and Bytes never splits
            // inside of UTF-8 bytes sequences thus `bytes` is still valid
            // UTF-8.
            unsafe { super::from_utf8_unchecked(bytes.as_bytes()) }
        })
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for a MultiCharEq wrapper
/////////////////////////////////////////////////////////////////////////////

#[doc(hidden)]
trait MultiCharEq {
    fn matches(&mut self, c: char) -> bool;
}

impl<F> MultiCharEq for F
where
    F: FnMut(char) -> bool,
{
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        (*self)(c)
    }
}

impl<const N: usize> MultiCharEq for [char; N] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&m| m == c)
    }
}

impl<const N: usize> MultiCharEq for &[char; N] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&m| m == c)
    }
}

impl MultiCharEq for &[char] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.iter().any(|&m| m == c)
    }
}

struct MultiCharEqPattern<C: MultiCharEq>(C);

#[derive(Clone, Debug)]
struct MultiCharEqSearcher<'a, C: MultiCharEq> {
    char_eq: C,
    haystack: &'a str,
    char_indices: super::CharIndices<'a>,
}

impl<'a, C: MultiCharEq> Pattern<&'a str> for MultiCharEqPattern<C> {
    type Searcher = MultiCharEqSearcher<'a, C>;

    #[inline]
    fn into_searcher(self, haystack: &'a str) -> MultiCharEqSearcher<'a, C> {
        MultiCharEqSearcher { haystack, char_eq: self.0, char_indices: haystack.char_indices() }
    }
}

unsafe impl<'a, C: MultiCharEq> Searcher<&'a str> for MultiCharEqSearcher<'a, C> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        let s = &mut self.char_indices;
        // Compare lengths of the internal byte slice iterator
        // to find length of current char
        let pre_len = s.iter.iter.len();
        if let Some((i, c)) = s.next() {
            let len = s.iter.iter.len();
            let char_len = pre_len - len;
            if self.char_eq.matches(c) {
                return SearchStep::Match(i, i + char_len);
            } else {
                return SearchStep::Reject(i, i + char_len);
            }
        }
        SearchStep::Done
    }
}

unsafe impl<'a, C: MultiCharEq> ReverseSearcher<&'a str> for MultiCharEqSearcher<'a, C> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        let s = &mut self.char_indices;
        // Compare lengths of the internal byte slice iterator
        // to find length of current char
        let pre_len = s.iter.iter.len();
        if let Some((i, c)) = s.next_back() {
            let len = s.iter.iter.len();
            let char_len = pre_len - len;
            if self.char_eq.matches(c) {
                return SearchStep::Match(i, i + char_len);
            } else {
                return SearchStep::Reject(i, i + char_len);
            }
        }
        SearchStep::Done
    }
}

impl<'a, C: MultiCharEq> DoubleEndedSearcher<&'a str> for MultiCharEqSearcher<'a, C> {}

/////////////////////////////////////////////////////////////////////////////

macro_rules! pattern_methods {
    ($t:ty, $pmap:expr, $smap:expr) => {
        type Searcher = $t;

        #[inline]
        fn into_searcher(self, haystack: &'a str) -> $t {
            ($smap)(($pmap)(self).into_searcher(haystack))
        }

        #[inline]
        fn is_contained_in(self, haystack: &'a str) -> bool {
            ($pmap)(self).is_contained_in(haystack)
        }

        #[inline]
        fn is_prefix_of(self, haystack: &'a str) -> bool {
            ($pmap)(self).is_prefix_of(haystack)
        }

        #[inline]
        fn strip_prefix_of(self, haystack: &'a str) -> Option<&'a str> {
            ($pmap)(self).strip_prefix_of(haystack)
        }

        #[inline]
        fn is_suffix_of(self, haystack: &'a str) -> bool
        where
            $t: ReverseSearcher<&'a str>,
        {
            ($pmap)(self).is_suffix_of(haystack)
        }

        #[inline]
        fn strip_suffix_of(self, haystack: &'a str) -> Option<&'a str>
        where
            $t: ReverseSearcher<&'a str>,
        {
            ($pmap)(self).strip_suffix_of(haystack)
        }
    };
}

macro_rules! searcher_methods {
    (forward) => {
        #[inline]
        fn haystack(&self) -> &'a str {
            self.0.haystack()
        }
        #[inline]
        fn next(&mut self) -> SearchStep {
            self.0.next()
        }
        #[inline]
        fn next_match(&mut self) -> Option<(usize, usize)> {
            self.0.next_match()
        }
        #[inline]
        fn next_reject(&mut self) -> Option<(usize, usize)> {
            self.0.next_reject()
        }
    };
    (reverse) => {
        #[inline]
        fn next_back(&mut self) -> SearchStep {
            self.0.next_back()
        }
        #[inline]
        fn next_match_back(&mut self) -> Option<(usize, usize)> {
            self.0.next_match_back()
        }
        #[inline]
        fn next_reject_back(&mut self) -> Option<(usize, usize)> {
            self.0.next_reject_back()
        }
    };
}

/// Associated type for `<[char; N] as Pattern<&'a str>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharArraySearcher<'a, const N: usize>(
    <MultiCharEqPattern<[char; N]> as Pattern<&'a str>>::Searcher,
);

/// Associated type for `<&[char; N] as Pattern<&'a str>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharArrayRefSearcher<'a, 'b, const N: usize>(
    <MultiCharEqPattern<&'b [char; N]> as Pattern<&'a str>>::Searcher,
);

/// Searches for chars that are equal to any of the [`char`]s in the array.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(['l', 'l']), Some(2));
/// assert_eq!("Hello world".find(['l', 'l']), Some(2));
/// ```
impl<'a, const N: usize> Pattern<&'a str> for [char; N] {
    pattern_methods!(CharArraySearcher<'a, N>, MultiCharEqPattern, CharArraySearcher);
}

unsafe impl<'a, const N: usize> Searcher<&'a str> for CharArraySearcher<'a, N> {
    searcher_methods!(forward);
}

unsafe impl<'a, const N: usize> ReverseSearcher<&'a str> for CharArraySearcher<'a, N> {
    searcher_methods!(reverse);
}

/// Searches for chars that are equal to any of the [`char`]s in the array.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(&['l', 'l']), Some(2));
/// assert_eq!("Hello world".find(&['l', 'l']), Some(2));
/// ```
impl<'a, 'b, const N: usize> Pattern<&'a str> for &'b [char; N] {
    pattern_methods!(CharArrayRefSearcher<'a, 'b, N>, MultiCharEqPattern, CharArrayRefSearcher);
}

unsafe impl<'a, 'b, const N: usize> Searcher<&'a str> for CharArrayRefSearcher<'a, 'b, N> {
    searcher_methods!(forward);
}

unsafe impl<'a, 'b, const N: usize> ReverseSearcher<&'a str> for CharArrayRefSearcher<'a, 'b, N> {
    searcher_methods!(reverse);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &[char]
/////////////////////////////////////////////////////////////////////////////

// Todo: Change / Remove due to ambiguity in meaning.

/// Associated type for `<&[char] as Pattern<&'a str>>::Searcher`.
#[derive(Clone, Debug)]
pub struct CharSliceSearcher<'a, 'b>(
    <MultiCharEqPattern<&'b [char]> as Pattern<&'a str>>::Searcher,
);

unsafe impl<'a, 'b> Searcher<&'a str> for CharSliceSearcher<'a, 'b> {
    searcher_methods!(forward);
}

unsafe impl<'a, 'b> ReverseSearcher<&'a str> for CharSliceSearcher<'a, 'b> {
    searcher_methods!(reverse);
}

impl<'a, 'b> DoubleEndedSearcher<&'a str> for CharSliceSearcher<'a, 'b> {}

/// Searches for chars that are equal to any of the [`char`]s in the slice.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(&['l', 'l'] as &[_]), Some(2));
/// assert_eq!("Hello world".find(&['l', 'l'][..]), Some(2));
/// ```
impl<'a, 'b> Pattern<&'a str> for &'b [char] {
    pattern_methods!(CharSliceSearcher<'a, 'b>, MultiCharEqPattern, CharSliceSearcher);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for F: FnMut(char) -> bool
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<F as Pattern<&'a str>>::Searcher`.
#[derive(Clone)]
pub struct CharPredicateSearcher<'a, F>(<MultiCharEqPattern<F> as Pattern<&'a str>>::Searcher)
where
    F: FnMut(char) -> bool;

impl<F> fmt::Debug for CharPredicateSearcher<'_, F>
where
    F: FnMut(char) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharPredicateSearcher")
            .field("haystack", &self.0.haystack)
            .field("char_indices", &self.0.char_indices)
            .finish()
    }
}
unsafe impl<'a, F> Searcher<&'a str> for CharPredicateSearcher<'a, F>
where
    F: FnMut(char) -> bool,
{
    searcher_methods!(forward);
}

unsafe impl<'a, F> ReverseSearcher<&'a str> for CharPredicateSearcher<'a, F>
where
    F: FnMut(char) -> bool,
{
    searcher_methods!(reverse);
}

impl<'a, F> DoubleEndedSearcher<&'a str> for CharPredicateSearcher<'a, F> where
    F: FnMut(char) -> bool
{
}

/// Searches for [`char`]s that match the given predicate.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(char::is_uppercase), Some(0));
/// assert_eq!("Hello world".find(|c| "aeiou".contains(c)), Some(1));
/// ```
impl<'a, F> Pattern<&'a str> for F
where
    F: FnMut(char) -> bool,
{
    pattern_methods!(CharPredicateSearcher<'a, F>, MultiCharEqPattern, CharPredicateSearcher);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &&str
/////////////////////////////////////////////////////////////////////////////

/// Delegates to the `&str` impl.
impl<'a, 'b, 'c> Pattern<&'a str> for &'c &'b str {
    pattern_methods!(StrSearcher<'a, 'b>, |&s| s, |s| s);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &str
/////////////////////////////////////////////////////////////////////////////

/// Non-allocating substring search.
///
/// Will handle the pattern `""` as returning empty matches at each character
/// boundary.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find("world"), Some(6));
/// ```
impl<'a, 'b> Pattern<&'a str> for &'b str {
    type Searcher = StrSearcher<'a, 'b>;

    #[inline]
    fn into_searcher(self, haystack: &'a str) -> StrSearcher<'a, 'b> {
        StrSearcher::new(haystack, self)
    }

    /// Checks whether the pattern matches at the front of the haystack.
    #[inline]
    fn is_prefix_of(self, haystack: &'a str) -> bool {
        haystack.as_bytes().starts_with(self.as_bytes())
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &'a str) -> bool {
        if self.len() == 0 {
            return true;
        }

        match self.len().cmp(&haystack.len()) {
            Ordering::Less => {
                if self.len() == 1 {
                    return haystack.as_bytes().contains(&self.as_bytes()[0]);
                }

                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                if self.len() <= 32 {
                    if let Some(result) = simd_contains(self, haystack) {
                        return result;
                    }
                }

                self.into_searcher(haystack).next_match().is_some()
            }
            _ => self == haystack,
        }
    }

    /// Removes the pattern from the front of haystack, if it matches.
    #[inline]
    fn strip_prefix_of(self, haystack: &'a str) -> Option<&'a str> {
        if self.is_prefix_of(haystack) {
            // SAFETY: prefix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(self.as_bytes().len()..)) }
        } else {
            None
        }
    }

    /// Checks whether the pattern matches at the back of the haystack.
    #[inline]
    fn is_suffix_of(self, haystack: &'a str) -> bool {
        haystack.as_bytes().ends_with(self.as_bytes())
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of(self, haystack: &'a str) -> Option<&'a str> {
        if self.is_suffix_of(haystack) {
            let i = haystack.len() - self.as_bytes().len();
            // SAFETY: suffix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(..i)) }
        } else {
            None
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Two Way substring searcher
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
/// Associated type for `<&str as Pattern<&'a str>>::Searcher`.
pub struct StrSearcher<'a, 'b>(crate::str_bytes::StrSearcher<'a, 'b>);

impl<'a, 'b> StrSearcher<'a, 'b> {
    fn new(haystack: &'a str, needle: &'b str) -> StrSearcher<'a, 'b> {
        let haystack = crate::str_bytes::Bytes::from(haystack);
        Self(crate::str_bytes::StrSearcher::new(haystack, needle))
    }
}

unsafe impl<'a, 'b> Searcher<&'a str> for StrSearcher<'a, 'b> {
    #[inline]
    fn haystack(&self) -> &'a str {
        let bytes = self.0.haystack().as_bytes();
        // SAFETY: self.0.haystack() was created from a &str.
        unsafe { crate::str::from_utf8_unchecked(bytes) }
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        self.0.next()
    }

    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        self.0.next_match()
    }

    fn next_reject(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject()
    }
}

unsafe impl<'a, 'b> ReverseSearcher<&'a str> for StrSearcher<'a, 'b> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        self.0.next_back()
    }

    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_match_back()
    }

    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        self.0.next_reject_back()
    }
}

/// SIMD search for short needles based on
/// Wojciech Muła's "SIMD-friendly algorithms for substring searching"[0]
///
/// It skips ahead by the vector width on each iteration (rather than the needle length as two-way
/// does) by probing the first and last byte of the needle for the whole vector width
/// and only doing full needle comparisons when the vectorized probe indicated potential matches.
///
/// Since the x86_64 baseline only offers SSE2 we only use u8x16 here.
/// If we ever ship std with for x86-64-v3 or adapt this for other platforms then wider vectors
/// should be evaluated.
///
/// For haystacks smaller than vector-size + needle length it falls back to
/// a naive O(n*m) search so this implementation should not be called on larger needles.
///
/// [0]: http://0x80.pl/articles/simd-strfind.html#sse-avx2
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn simd_contains(needle: &str, haystack: &str) -> Option<bool> {
    let needle = needle.as_bytes();
    let haystack = haystack.as_bytes();

    debug_assert!(needle.len() > 1);

    use crate::ops::BitAnd;
    use crate::simd::mask8x16 as Mask;
    use crate::simd::u8x16 as Block;
    use crate::simd::{SimdPartialEq, ToBitMask};

    let first_probe = needle[0];
    let last_byte_offset = needle.len() - 1;

    // the offset used for the 2nd vector
    let second_probe_offset = if needle.len() == 2 {
        // never bail out on len=2 needles because the probes will fully cover them and have
        // no degenerate cases.
        1
    } else {
        // try a few bytes in case first and last byte of the needle are the same
        let Some(second_probe_offset) = (needle.len().saturating_sub(4)..needle.len()).rfind(|&idx| needle[idx] != first_probe) else {
            // fall back to other search methods if we can't find any different bytes
            // since we could otherwise hit some degenerate cases
            return None;
        };
        second_probe_offset
    };

    // do a naive search if the haystack is too small to fit
    if haystack.len() < Block::LANES + last_byte_offset {
        return Some(haystack.windows(needle.len()).any(|c| c == needle));
    }

    let first_probe: Block = Block::splat(first_probe);
    let second_probe: Block = Block::splat(needle[second_probe_offset]);
    // first byte are already checked by the outer loop. to verify a match only the
    // remainder has to be compared.
    let trimmed_needle = &needle[1..];

    // this #[cold] is load-bearing, benchmark before removing it...
    let check_mask = #[cold]
    |idx, mask: u16, skip: bool| -> bool {
        if skip {
            return false;
        }

        // and so is this. optimizations are weird.
        let mut mask = mask;

        while mask != 0 {
            let trailing = mask.trailing_zeros();
            let offset = idx + trailing as usize + 1;
            // SAFETY: mask is between 0 and 15 trailing zeroes, we skip one additional byte that was already compared
            // and then take trimmed_needle.len() bytes. This is within the bounds defined by the outer loop
            unsafe {
                let sub = haystack.get_unchecked(offset..).get_unchecked(..trimmed_needle.len());
                if small_slice_eq(sub, trimmed_needle) {
                    return true;
                }
            }
            mask &= !(1 << trailing);
        }
        return false;
    };

    let test_chunk = |idx| -> u16 {
        // SAFETY: this requires at least LANES bytes being readable at idx
        // that is ensured by the loop ranges (see comments below)
        let a: Block = unsafe { haystack.as_ptr().add(idx).cast::<Block>().read_unaligned() };
        // SAFETY: this requires LANES + block_offset bytes being readable at idx
        let b: Block = unsafe {
            haystack.as_ptr().add(idx).add(second_probe_offset).cast::<Block>().read_unaligned()
        };
        let eq_first: Mask = a.simd_eq(first_probe);
        let eq_last: Mask = b.simd_eq(second_probe);
        let both = eq_first.bitand(eq_last);
        let mask = both.to_bitmask();

        return mask;
    };

    let mut i = 0;
    let mut result = false;
    // The loop condition must ensure that there's enough headroom to read LANE bytes,
    // and not only at the current index but also at the index shifted by block_offset
    const UNROLL: usize = 4;
    while i + last_byte_offset + UNROLL * Block::LANES < haystack.len() && !result {
        let mut masks = [0u16; UNROLL];
        for j in 0..UNROLL {
            masks[j] = test_chunk(i + j * Block::LANES);
        }
        for j in 0..UNROLL {
            let mask = masks[j];
            if mask != 0 {
                result |= check_mask(i + j * Block::LANES, mask, result);
            }
        }
        i += UNROLL * Block::LANES;
    }
    while i + last_byte_offset + Block::LANES < haystack.len() && !result {
        let mask = test_chunk(i);
        if mask != 0 {
            result |= check_mask(i, mask, result);
        }
        i += Block::LANES;
    }

    // Process the tail that didn't fit into LANES-sized steps.
    // This simply repeats the same procedure but as right-aligned chunk instead
    // of a left-aligned one. The last byte must be exactly flush with the string end so
    // we don't miss a single byte or read out of bounds.
    let i = haystack.len() - last_byte_offset - Block::LANES;
    let mask = test_chunk(i);
    if mask != 0 {
        result |= check_mask(i, mask, result);
    }

    Some(result)
}

/// Compares short slices for equality.
///
/// It avoids a call to libc's memcmp which is faster on long slices
/// due to SIMD optimizations but it incurs a function call overhead.
///
/// # Safety
///
/// Both slices must have the same length.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] // only called on x86
#[inline]
unsafe fn small_slice_eq(x: &[u8], y: &[u8]) -> bool {
    debug_assert_eq!(x.len(), y.len());
    // This function is adapted from
    // https://github.com/BurntSushi/memchr/blob/8037d11b4357b0f07be2bb66dc2659d9cf28ad32/src/memmem/util.rs#L32

    // If we don't have enough bytes to do 4-byte at a time loads, then
    // fall back to the naive slow version.
    //
    // Potential alternative: We could do a copy_nonoverlapping combined with a mask instead
    // of a loop. Benchmark it.
    if x.len() < 4 {
        for (&b1, &b2) in x.iter().zip(y) {
            if b1 != b2 {
                return false;
            }
        }
        return true;
    }
    // When we have 4 or more bytes to compare, then proceed in chunks of 4 at
    // a time using unaligned loads.
    //
    // Also, why do 4 byte loads instead of, say, 8 byte loads? The reason is
    // that this particular version of memcmp is likely to be called with tiny
    // needles. That means that if we do 8 byte loads, then a higher proportion
    // of memcmp calls will use the slower variant above. With that said, this
    // is a hypothesis and is only loosely supported by benchmarks. There's
    // likely some improvement that could be made here. The main thing here
    // though is to optimize for latency, not throughput.

    // SAFETY: Via the conditional above, we know that both `px` and `py`
    // have the same length, so `px < pxend` implies that `py < pyend`.
    // Thus, dereferencing both `px` and `py` in the loop below is safe.
    //
    // Moreover, we set `pxend` and `pyend` to be 4 bytes before the actual
    // end of `px` and `py`. Thus, the final dereference outside of the
    // loop is guaranteed to be valid. (The final comparison will overlap with
    // the last comparison done in the loop for lengths that aren't multiples
    // of four.)
    //
    // Finally, we needn't worry about alignment here, since we do unaligned
    // loads.
    unsafe {
        let (mut px, mut py) = (x.as_ptr(), y.as_ptr());
        let (pxend, pyend) = (px.add(x.len() - 4), py.add(y.len() - 4));
        while px < pxend {
            let vx = (px as *const u32).read_unaligned();
            let vy = (py as *const u32).read_unaligned();
            if vx != vy {
                return false;
            }
            px = px.add(4);
            py = py.add(4);
        }
        let vx = (pxend as *const u32).read_unaligned();
        let vy = (pyend as *const u32).read_unaligned();
        vx == vy
    }
}
