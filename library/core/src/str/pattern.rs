//! The string Pattern API.
//!
//! The Pattern API provides a generic mechanism for using different pattern
//! types when searching through a string.
//!
//! For more details, see the traits [`Pattern`], [`Searcher`],
//! [`ReverseSearcher`], and [`DoubleEndedSearcher`].
//!
//! Although this API is unstable, it is exposed via stable APIs on the
//! [`str`] type.
//!
//! # Examples
//!
//! [`Pattern`] is [implemented][pattern-impls] in the stable API for
//! [`&str`][`str`], [`char`], slices of [`char`], and functions and closures
//! implementing `FnMut(char) -> bool`.
//!
//! ```
//! let s = "Can you find a needle in a haystack?";
//!
//! // &str pattern
//! assert_eq!(s.find("you"), Some(4));
//! // char pattern
//! assert_eq!(s.find('n'), Some(2));
//! // array of chars pattern
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u']), Some(1));
//! // slice of chars pattern
//! assert_eq!(s.find(&['a', 'e', 'i', 'o', 'u'][..]), Some(1));
//! // closure pattern
//! assert_eq!(s.find(|c: char| c.is_ascii_punctuation()), Some(35));
//! ```
//!
//! [pattern-impls]: Pattern#implementors

#![unstable(
    feature = "pattern",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "27721"
)]

use crate::char::MAX_LEN_UTF8;
use crate::cmp::Ordering;
use crate::convert::TryInto as _;
use crate::slice::byte_pattern::{self, MatchOnly, RejectAndMatch, TwoWaySearcher};
use crate::slice::memchr;
use crate::{cmp, fmt};

// Pattern

/// A string pattern.
///
/// A `Pattern` expresses that the implementing type
/// can be used as a string pattern for searching in a [`&str`][str].
///
/// For example, both `'a'` and `"aa"` are patterns that
/// would match at index `1` in the string `"baaaab"`.
///
/// The trait itself acts as a builder for an associated
/// [`Searcher`] type, which does the actual work of finding
/// occurrences of the pattern in a string.
///
/// Depending on the type of the pattern, the behavior of methods like
/// [`str::find`] and [`str::contains`] can change. The table below describes
/// some of those behaviors.
///
/// | Pattern type             | Match condition                           |
/// |--------------------------|-------------------------------------------|
/// | `&str`                   | is substring                              |
/// | `char`                   | is contained in string                    |
/// | `&[char]`                | any char in slice is contained in string  |
/// | `F: FnMut(char) -> bool` | `F` returns `true` for a char in string   |
/// | `&&str`                  | is substring                              |
/// | `&String`                | is substring                              |
///
/// # Examples
///
/// ```
/// // &str
/// assert_eq!("abaaa".find("ba"), Some(1));
/// assert_eq!("abaaa".find("bac"), None);
///
/// // char
/// assert_eq!("abaaa".find('a'), Some(0));
/// assert_eq!("abaaa".find('b'), Some(1));
/// assert_eq!("abaaa".find('c'), None);
///
/// // &[char; N]
/// assert_eq!("ab".find(&['b', 'a']), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z']), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd']), None);
///
/// // &[char]
/// assert_eq!("ab".find(&['b', 'a'][..]), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z'][..]), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd'][..]), None);
///
/// // FnMut(char) -> bool
/// assert_eq!("abcdef_z".find(|ch| ch > 'd' && ch < 'y'), Some(4));
/// assert_eq!("abcddd_z".find(|ch| ch > 'd' && ch < 'y'), None);
/// ```
pub trait Pattern: Sized {
    /// Associated searcher for this pattern
    type Searcher<'a>: Searcher<'a>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: &str) -> Self::Searcher<'_>;

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &str) -> bool {
        self.into_searcher(haystack).next_match().is_some()
    }

    /// Checks whether the pattern matches at the front of the haystack
    #[inline]
    fn is_prefix_of(self, haystack: &str) -> bool {
        matches!(self.into_searcher(haystack).next(), SearchStep::Match(0, _))
    }

    /// Checks whether the pattern matches at the back of the haystack
    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a str) -> bool
    where
        Self::Searcher<'a>: ReverseSearcher<'a>,
    {
        matches!(self.into_searcher(haystack).next_back(), SearchStep::Match(_, j) if haystack.len() == j)
    }

    /// Removes the pattern from the front of haystack, if it matches.
    #[inline]
    fn strip_prefix_of(self, haystack: &str) -> Option<&str> {
        if let SearchStep::Match(start, len) = self.into_searcher(haystack).next() {
            debug_assert_eq!(
                start, 0,
                "The first search step from Searcher \
                 must include the first character"
            );
            // SAFETY: `Searcher` is known to return valid indices.
            unsafe { Some(haystack.get_unchecked(len..)) }
        } else {
            None
        }
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a str) -> Option<&'a str>
    where
        Self::Searcher<'a>: ReverseSearcher<'a>,
    {
        if let SearchStep::Match(start, end) = self.into_searcher(haystack).next_back() {
            debug_assert_eq!(
                end,
                haystack.len(),
                "The first search step from ReverseSearcher \
                 must include the last character"
            );
            // SAFETY: `Searcher` is known to return valid indices.
            unsafe { Some(haystack.get_unchecked(..start)) }
        } else {
            None
        }
    }

    /// Returns the pattern as utf-8 bytes if possible.
    fn as_utf8_pattern(&self) -> Option<Utf8Pattern<'_>> {
        None
    }
}
/// Result of calling [`Pattern::as_utf8_pattern()`].
/// Can be used for inspecting the contents of a [`Pattern`] in cases
/// where the underlying representation can be represented as UTF-8.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Utf8Pattern<'a> {
    /// Type returned by String and str types.
    StringPattern(&'a [u8]),
    /// Type returned by char types.
    CharPattern(char),
}

// Searcher

/// Result of calling [`Searcher::next()`] or [`ReverseSearcher::next_back()`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SearchStep {
    /// Expresses that a match of the pattern has been found at
    /// `haystack[a..b]`.
    Match(usize, usize),
    /// Expresses that `haystack[a..b]` has been rejected as a possible match
    /// of the pattern.
    ///
    /// Note that there might be more than one `Reject` between two `Match`es,
    /// there is no requirement for them to be combined into one.
    Reject(usize, usize),
    /// Expresses that every byte of the haystack has been visited, ending
    /// the iteration.
    Done,
}

/// A searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping
/// matches of a pattern starting from the front (left) of a string.
///
/// It will be implemented by associated `Searcher`
/// types of the [`Pattern`] trait.
///
/// The trait is marked unsafe because the indices returned by the
/// [`next()`][Searcher::next] methods are required to lie on valid utf8
/// boundaries in the haystack. This enables consumers of this trait to
/// slice the haystack without additional runtime checks.
pub unsafe trait Searcher<'a> {
    /// Getter for the underlying string to be searched in
    ///
    /// Will always return the same [`&str`][str].
    fn haystack(&self) -> &'a str;

    /// Performs the next search step starting from the front.
    ///
    /// - Returns [`Match(a, b)`][SearchStep::Match] if `haystack[a..b]` matches
    ///   the pattern.
    /// - Returns [`Reject(a, b)`][SearchStep::Reject] if `haystack[a..b]` can
    ///   not match the pattern, even partially.
    /// - Returns [`Done`][SearchStep::Done] if every byte of the haystack has
    ///   been visited.
    ///
    /// The stream of [`Match`][SearchStep::Match] and
    /// [`Reject`][SearchStep::Reject] values up to a [`Done`][SearchStep::Done]
    /// will contain index ranges that are adjacent, non-overlapping,
    /// covering the whole haystack, and laying on utf8 boundaries.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero length.
    ///
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"`
    /// might produce the stream
    /// `[Reject(0, 1), Reject(1, 2), Match(2, 5), Reject(5, 8)]`
    fn next(&mut self) -> SearchStep;

    /// Finds the next [`Match`][SearchStep::Match] result. See [`next()`][Searcher::next].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the returned ranges
    /// of this and [`next_reject`][Searcher::next_reject] will overlap. This will return
    /// `(start_match, end_match)`, where start_match is the index of where
    /// the match begins, and end_match is the index after the end of the match.
    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.next() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result. See [`next()`][Searcher::next]
    /// and [`next_match()`][Searcher::next_match].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the returned ranges
    /// of this and [`next_match`][Searcher::next_match] will overlap.
    #[inline]
    fn next_reject(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.next() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

/// A reverse searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping
/// matches of a pattern starting from the back (right) of a string.
///
/// It will be implemented by associated [`Searcher`]
/// types of the [`Pattern`] trait if the pattern supports searching
/// for it from the back.
///
/// The index ranges returned by this trait are not required
/// to exactly match those of the forward search in reverse.
///
/// For the reason why this trait is marked unsafe, see the
/// parent trait [`Searcher`].
pub unsafe trait ReverseSearcher<'a>: Searcher<'a> {
    /// Performs the next search step starting from the back.
    ///
    /// - Returns [`Match(a, b)`][SearchStep::Match] if `haystack[a..b]`
    ///   matches the pattern.
    /// - Returns [`Reject(a, b)`][SearchStep::Reject] if `haystack[a..b]`
    ///   can not match the pattern, even partially.
    /// - Returns [`Done`][SearchStep::Done] if every byte of the haystack
    ///   has been visited
    ///
    /// The stream of [`Match`][SearchStep::Match] and
    /// [`Reject`][SearchStep::Reject] values up to a [`Done`][SearchStep::Done]
    /// will contain index ranges that are adjacent, non-overlapping,
    /// covering the whole haystack, and laying on utf8 boundaries.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero length.
    ///
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"`
    /// might produce the stream
    /// `[Reject(7, 8), Match(4, 7), Reject(1, 4), Reject(0, 1)]`.
    fn next_back(&mut self) -> SearchStep;

    /// Finds the next [`Match`][SearchStep::Match] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.next_back() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    #[inline]
    fn next_reject_back(&mut self) -> Option<(usize, usize)> {
        loop {
            match self.next_back() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

/// A marker trait to express that a [`ReverseSearcher`]
/// can be used for a [`DoubleEndedIterator`] implementation.
///
/// For this, the impl of [`Searcher`] and [`ReverseSearcher`] need
/// to follow these conditions:
///
/// - All results of `next()` need to be identical
///   to the results of `next_back()` in reverse order.
/// - `next()` and `next_back()` need to behave as
///   the two ends of a range of values, that is they
///   can not "walk past each other".
///
/// # Examples
///
/// `char::Searcher` is a `DoubleEndedSearcher` because searching for a
/// [`char`] only requires looking at one at a time, which behaves the same
/// from both ends.
///
/// `(&str)::Searcher` is not a `DoubleEndedSearcher` because
/// the pattern `"aa"` in the haystack `"aaa"` matches as either
/// `"[aa]a"` or `"a[aa]"`, depending on which side it is searched.
pub trait DoubleEndedSearcher<'a>: ReverseSearcher<'a> {}

/////////////////////////////////////////////////////////////////////////////
// Impl for char
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<char as Pattern>::Searcher<'a>`.
#[derive(Clone, Debug)]
pub struct CharSearcher<'a> {
    haystack: &'a str,
    // safety invariant: `finger`/`finger_back` must be a valid utf8 byte index of `haystack`
    // This invariant can be broken *within* next_match and next_match_back, however
    // they must exit with fingers on valid code point boundaries.
    /// `finger` is the current byte index of the forward search.
    /// Imagine that it exists before the byte at its index, i.e.
    /// `haystack[finger]` is the first byte of the slice we must inspect during
    /// forward searching
    finger: usize,
    /// `finger_back` is the current byte index of the reverse search.
    /// Imagine that it exists after the byte at its index, i.e.
    /// haystack[finger_back - 1] is the last byte of the slice we must inspect during
    /// forward searching (and thus the first byte to be inspected when calling next_back()).
    finger_back: usize,
    /// The character being searched for
    needle: char,

    // safety invariant: `utf8_size` must be less than 5
    /// The number of bytes `needle` takes up when encoded in utf8.
    utf8_size: u8,
    /// A utf8 encoded copy of the `needle`
    utf8_encoded: [u8; 4],
}

impl CharSearcher<'_> {
    fn utf8_size(&self) -> usize {
        self.utf8_size.into()
    }
}

unsafe impl<'a> Searcher<'a> for CharSearcher<'a> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.haystack
    }
    #[inline]
    fn next(&mut self) -> SearchStep {
        let old_finger = self.finger;
        // SAFETY: 1-4 guarantee safety of `get_unchecked`
        // 1. `self.finger` and `self.finger_back` are kept on unicode boundaries
        //    (this is invariant)
        // 2. `self.finger >= 0` since it starts at 0 and only increases
        // 3. `self.finger < self.finger_back` because otherwise the char `iter`
        //    would return `SearchStep::Done`
        // 4. `self.finger` comes before the end of the haystack because `self.finger_back`
        //    starts at the end and only decreases
        let slice = unsafe { self.haystack.get_unchecked(old_finger..self.finger_back) };
        let mut iter = slice.chars();
        let old_len = iter.iter.len();
        if let Some(ch) = iter.next() {
            // add byte offset of current character
            // without re-encoding as utf-8
            self.finger += old_len - iter.iter.len();
            if ch == self.needle {
                SearchStep::Match(old_finger, self.finger)
            } else {
                SearchStep::Reject(old_finger, self.finger)
            }
        } else {
            SearchStep::Done
        }
    }
    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        loop {
            // get the haystack after the last character found
            let bytes = self.haystack.as_bytes().get(self.finger..self.finger_back)?;
            // the last byte of the utf8 encoded needle
            // SAFETY: we have an invariant that `utf8_size < 5`
            let last_byte = unsafe { *self.utf8_encoded.get_unchecked(self.utf8_size() - 1) };
            if let Some(index) = memchr::memchr(last_byte, bytes) {
                // The new finger is the index of the byte we found,
                // plus one, since we memchr'd for the last byte of the character.
                //
                // Note that this doesn't always give us a finger on a UTF8 boundary.
                // If we *didn't* find our character
                // we may have indexed to the non-last byte of a 3-byte or 4-byte character.
                // We can't just skip to the next valid starting byte because a character like
                // ꁁ (U+A041 YI SYLLABLE PA), utf-8 `EA 81 81` will have us always find
                // the second byte when searching for the third.
                //
                // However, this is totally okay. While we have the invariant that
                // self.finger is on a UTF8 boundary, this invariant is not relied upon
                // within this method (it is relied upon in CharSearcher::next()).
                //
                // We only exit this method when we reach the end of the string, or if we
                // find something. When we find something the `finger` will be set
                // to a UTF8 boundary.
                self.finger += index + 1;
                if self.finger >= self.utf8_size() {
                    let found_char = self.finger - self.utf8_size();
                    if let Some(slice) = self.haystack.as_bytes().get(found_char..self.finger) {
                        if slice == &self.utf8_encoded[0..self.utf8_size()] {
                            return Some((found_char, self.finger));
                        }
                    }
                }
            } else {
                // found nothing, exit
                self.finger = self.finger_back;
                return None;
            }
        }
    }

    // let next_reject use the default implementation from the Searcher trait
}

unsafe impl<'a> ReverseSearcher<'a> for CharSearcher<'a> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        let old_finger = self.finger_back;
        // SAFETY: see the comment for next() above
        let slice = unsafe { self.haystack.get_unchecked(self.finger..old_finger) };
        let mut iter = slice.chars();
        let old_len = iter.iter.len();
        if let Some(ch) = iter.next_back() {
            // subtract byte offset of current character
            // without re-encoding as utf-8
            self.finger_back -= old_len - iter.iter.len();
            if ch == self.needle {
                SearchStep::Match(self.finger_back, old_finger)
            } else {
                SearchStep::Reject(self.finger_back, old_finger)
            }
        } else {
            SearchStep::Done
        }
    }
    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        let haystack = self.haystack.as_bytes();
        loop {
            // get the haystack up to but not including the last character searched
            let bytes = haystack.get(self.finger..self.finger_back)?;
            // the last byte of the utf8 encoded needle
            // SAFETY: we have an invariant that `utf8_size < 5`
            let last_byte = unsafe { *self.utf8_encoded.get_unchecked(self.utf8_size() - 1) };
            if let Some(index) = memchr::memrchr(last_byte, bytes) {
                // we searched a slice that was offset by self.finger,
                // add self.finger to recoup the original index
                let index = self.finger + index;
                // memrchr will return the index of the byte we wish to
                // find. In case of an ASCII character, this is indeed
                // were we wish our new finger to be ("after" the found
                // char in the paradigm of reverse iteration). For
                // multibyte chars we need to skip down by the number of more
                // bytes they have than ASCII
                let shift = self.utf8_size() - 1;
                if index >= shift {
                    let found_char = index - shift;
                    if let Some(slice) = haystack.get(found_char..(found_char + self.utf8_size())) {
                        if slice == &self.utf8_encoded[0..self.utf8_size()] {
                            // move finger to before the character found (i.e., at its start index)
                            self.finger_back = found_char;
                            return Some((self.finger_back, self.finger_back + self.utf8_size()));
                        }
                    }
                }
                // We can't use finger_back = index - size + 1 here. If we found the last char
                // of a different-sized character (or the middle byte of a different character)
                // we need to bump the finger_back down to `index`. This similarly makes
                // `finger_back` have the potential to no longer be on a boundary,
                // but this is OK since we only exit this function on a boundary
                // or when the haystack has been searched completely.
                //
                // Unlike next_match this does not
                // have the problem of repeated bytes in utf-8 because
                // we're searching for the last byte, and we can only have
                // found the last byte when searching in reverse.
                self.finger_back = index;
            } else {
                self.finger_back = self.finger;
                // found nothing, exit
                return None;
            }
        }
    }

    // let next_reject_back use the default implementation from the Searcher trait
}

impl<'a> DoubleEndedSearcher<'a> for CharSearcher<'a> {}

/// Searches for chars that are equal to a given [`char`].
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find('o'), Some(4));
/// ```
impl Pattern for char {
    type Searcher<'a> = CharSearcher<'a>;

    #[inline]
    fn into_searcher<'a>(self, haystack: &'a str) -> Self::Searcher<'a> {
        let mut utf8_encoded = [0; MAX_LEN_UTF8];
        let utf8_size = self
            .encode_utf8(&mut utf8_encoded)
            .len()
            .try_into()
            .expect("char len should be less than 255");

        CharSearcher {
            haystack,
            finger: 0,
            finger_back: haystack.len(),
            needle: self,
            utf8_size,
            utf8_encoded,
        }
    }

    #[inline]
    fn is_contained_in(self, haystack: &str) -> bool {
        if (self as u32) < 128 {
            haystack.as_bytes().contains(&(self as u8))
        } else {
            let mut buffer = [0u8; 4];
            self.encode_utf8(&mut buffer).is_contained_in(haystack)
        }
    }

    #[inline]
    fn is_prefix_of(self, haystack: &str) -> bool {
        self.encode_utf8(&mut [0u8; 4]).is_prefix_of(haystack)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &str) -> Option<&str> {
        self.encode_utf8(&mut [0u8; 4]).strip_prefix_of(haystack)
    }

    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a str) -> bool
    where
        Self::Searcher<'a>: ReverseSearcher<'a>,
    {
        self.encode_utf8(&mut [0u8; 4]).is_suffix_of(haystack)
    }

    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a str) -> Option<&'a str>
    where
        Self::Searcher<'a>: ReverseSearcher<'a>,
    {
        self.encode_utf8(&mut [0u8; 4]).strip_suffix_of(haystack)
    }

    #[inline]
    fn as_utf8_pattern(&self) -> Option<Utf8Pattern<'_>> {
        Some(Utf8Pattern::CharPattern(*self))
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
        self.contains(&c)
    }
}

impl<const N: usize> MultiCharEq for &[char; N] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.contains(&c)
    }
}

impl MultiCharEq for &[char] {
    #[inline]
    fn matches(&mut self, c: char) -> bool {
        self.contains(&c)
    }
}

struct MultiCharEqPattern<C: MultiCharEq>(C);

#[derive(Clone, Debug)]
struct MultiCharEqSearcher<'a, C: MultiCharEq> {
    char_eq: C,
    haystack: &'a str,
    char_indices: super::CharIndices<'a>,
}

impl<C: MultiCharEq> Pattern for MultiCharEqPattern<C> {
    type Searcher<'a> = MultiCharEqSearcher<'a, C>;

    #[inline]
    fn into_searcher(self, haystack: &str) -> MultiCharEqSearcher<'_, C> {
        MultiCharEqSearcher { haystack, char_eq: self.0, char_indices: haystack.char_indices() }
    }
}

unsafe impl<'a, C: MultiCharEq> Searcher<'a> for MultiCharEqSearcher<'a, C> {
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

unsafe impl<'a, C: MultiCharEq> ReverseSearcher<'a> for MultiCharEqSearcher<'a, C> {
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

impl<'a, C: MultiCharEq> DoubleEndedSearcher<'a> for MultiCharEqSearcher<'a, C> {}

/////////////////////////////////////////////////////////////////////////////

macro_rules! pattern_methods {
    ($a:lifetime, $t:ty, $pmap:expr, $smap:expr) => {
        type Searcher<$a> = $t;

        #[inline]
        fn into_searcher<$a>(self, haystack: &$a str) -> $t {
            ($smap)(($pmap)(self).into_searcher(haystack))
        }

        #[inline]
        fn is_contained_in<$a>(self, haystack: &$a str) -> bool {
            ($pmap)(self).is_contained_in(haystack)
        }

        #[inline]
        fn is_prefix_of<$a>(self, haystack: &$a str) -> bool {
            ($pmap)(self).is_prefix_of(haystack)
        }

        #[inline]
        fn strip_prefix_of<$a>(self, haystack: &$a str) -> Option<&$a str> {
            ($pmap)(self).strip_prefix_of(haystack)
        }

        #[inline]
        fn is_suffix_of<$a>(self, haystack: &$a str) -> bool
        where
            $t: ReverseSearcher<$a>,
        {
            ($pmap)(self).is_suffix_of(haystack)
        }

        #[inline]
        fn strip_suffix_of<$a>(self, haystack: &$a str) -> Option<&$a str>
        where
            $t: ReverseSearcher<$a>,
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

/// Associated type for `<[char; N] as Pattern>::Searcher<'a>`.
#[derive(Clone, Debug)]
pub struct CharArraySearcher<'a, const N: usize>(
    <MultiCharEqPattern<[char; N]> as Pattern>::Searcher<'a>,
);

/// Associated type for `<&[char; N] as Pattern>::Searcher<'a>`.
#[derive(Clone, Debug)]
pub struct CharArrayRefSearcher<'a, 'b, const N: usize>(
    <MultiCharEqPattern<&'b [char; N]> as Pattern>::Searcher<'a>,
);

/// Searches for chars that are equal to any of the [`char`]s in the array.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(['o', 'l']), Some(2));
/// assert_eq!("Hello world".find(['h', 'w']), Some(6));
/// ```
impl<const N: usize> Pattern for [char; N] {
    pattern_methods!('a, CharArraySearcher<'a, N>, MultiCharEqPattern, CharArraySearcher);
}

unsafe impl<'a, const N: usize> Searcher<'a> for CharArraySearcher<'a, N> {
    searcher_methods!(forward);
}

unsafe impl<'a, const N: usize> ReverseSearcher<'a> for CharArraySearcher<'a, N> {
    searcher_methods!(reverse);
}

impl<'a, const N: usize> DoubleEndedSearcher<'a> for CharArraySearcher<'a, N> {}

/// Searches for chars that are equal to any of the [`char`]s in the array.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(&['o', 'l']), Some(2));
/// assert_eq!("Hello world".find(&['h', 'w']), Some(6));
/// ```
impl<'b, const N: usize> Pattern for &'b [char; N] {
    pattern_methods!('a, CharArrayRefSearcher<'a, 'b, N>, MultiCharEqPattern, CharArrayRefSearcher);
}

unsafe impl<'a, 'b, const N: usize> Searcher<'a> for CharArrayRefSearcher<'a, 'b, N> {
    searcher_methods!(forward);
}

unsafe impl<'a, 'b, const N: usize> ReverseSearcher<'a> for CharArrayRefSearcher<'a, 'b, N> {
    searcher_methods!(reverse);
}

impl<'a, 'b, const N: usize> DoubleEndedSearcher<'a> for CharArrayRefSearcher<'a, 'b, N> {}

/////////////////////////////////////////////////////////////////////////////
// Impl for &[char]
/////////////////////////////////////////////////////////////////////////////

// Todo: Change / Remove due to ambiguity in meaning.

/// Associated type for `<&[char] as Pattern>::Searcher<'a>`.
#[derive(Clone, Debug)]
pub struct CharSliceSearcher<'a, 'b>(<MultiCharEqPattern<&'b [char]> as Pattern>::Searcher<'a>);

unsafe impl<'a, 'b> Searcher<'a> for CharSliceSearcher<'a, 'b> {
    searcher_methods!(forward);
}

unsafe impl<'a, 'b> ReverseSearcher<'a> for CharSliceSearcher<'a, 'b> {
    searcher_methods!(reverse);
}

impl<'a, 'b> DoubleEndedSearcher<'a> for CharSliceSearcher<'a, 'b> {}

/// Searches for chars that are equal to any of the [`char`]s in the slice.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(&['o', 'l'][..]), Some(2));
/// assert_eq!("Hello world".find(&['h', 'w'][..]), Some(6));
/// ```
impl<'b> Pattern for &'b [char] {
    pattern_methods!('a, CharSliceSearcher<'a, 'b>, MultiCharEqPattern, CharSliceSearcher);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for F: FnMut(char) -> bool
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<F as Pattern>::Searcher<'a>`.
#[derive(Clone)]
pub struct CharPredicateSearcher<'a, F>(<MultiCharEqPattern<F> as Pattern>::Searcher<'a>)
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
unsafe impl<'a, F> Searcher<'a> for CharPredicateSearcher<'a, F>
where
    F: FnMut(char) -> bool,
{
    searcher_methods!(forward);
}

unsafe impl<'a, F> ReverseSearcher<'a> for CharPredicateSearcher<'a, F>
where
    F: FnMut(char) -> bool,
{
    searcher_methods!(reverse);
}

impl<'a, F> DoubleEndedSearcher<'a> for CharPredicateSearcher<'a, F> where F: FnMut(char) -> bool {}

/// Searches for [`char`]s that match the given predicate.
///
/// # Examples
///
/// ```
/// assert_eq!("Hello world".find(char::is_uppercase), Some(0));
/// assert_eq!("Hello world".find(|c| "aeiou".contains(c)), Some(1));
/// ```
impl<F> Pattern for F
where
    F: FnMut(char) -> bool,
{
    pattern_methods!('a, CharPredicateSearcher<'a, F>, MultiCharEqPattern, CharPredicateSearcher);
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &&str
/////////////////////////////////////////////////////////////////////////////

/// Delegates to the `&str` impl.
impl<'b, 'c> Pattern for &'c &'b str {
    pattern_methods!('a, StrSearcher<'a, 'b>, |&s| s, |s| s);
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
impl<'b> Pattern for &'b str {
    type Searcher<'a> = StrSearcher<'a, 'b>;

    #[inline]
    fn into_searcher(self, haystack: &str) -> StrSearcher<'_, 'b> {
        StrSearcher::new(haystack, self)
    }

    /// Checks whether the pattern matches at the front of the haystack.
    #[inline]
    fn is_prefix_of(self, haystack: &str) -> bool {
        haystack.as_bytes().starts_with(self.as_bytes())
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &str) -> bool {
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
                    if let Some(result) =
                        byte_pattern::simd_contains(self.as_bytes(), haystack.as_bytes())
                    {
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
    fn strip_prefix_of(self, haystack: &str) -> Option<&str> {
        if self.is_prefix_of(haystack) {
            // SAFETY: prefix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(self.as_bytes().len()..)) }
        } else {
            None
        }
    }

    /// Checks whether the pattern matches at the back of the haystack.
    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a str) -> bool
    where
        Self::Searcher<'a>: ReverseSearcher<'a>,
    {
        haystack.as_bytes().ends_with(self.as_bytes())
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a str) -> Option<&'a str>
    where
        Self::Searcher<'a>: ReverseSearcher<'a>,
    {
        if self.is_suffix_of(haystack) {
            let i = haystack.len() - self.as_bytes().len();
            // SAFETY: suffix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(..i)) }
        } else {
            None
        }
    }

    #[inline]
    fn as_utf8_pattern(&self) -> Option<Utf8Pattern<'_>> {
        Some(Utf8Pattern::StringPattern(self.as_bytes()))
    }
}

/////////////////////////////////////////////////////////////////////////////
// Two Way substring searcher
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
/// Associated type for `<&str as Pattern>::Searcher<'a>`.
pub struct StrSearcher<'a, 'b> {
    haystack: &'a str,
    needle: &'b str,

    searcher: StrSearcherImpl,
}

#[derive(Clone, Debug)]
enum StrSearcherImpl {
    Empty(EmptyNeedle),
    TwoWay(TwoWaySearcher),
}

#[derive(Clone, Debug)]
struct EmptyNeedle {
    position: usize,
    end: usize,
    is_match_fw: bool,
    is_match_bw: bool,
    // Needed in case of an empty haystack, see #85462
    is_finished: bool,
}

impl<'a, 'b> StrSearcher<'a, 'b> {
    fn new(haystack: &'a str, needle: &'b str) -> StrSearcher<'a, 'b> {
        if needle.is_empty() {
            StrSearcher {
                haystack,
                needle,
                searcher: StrSearcherImpl::Empty(EmptyNeedle {
                    position: 0,
                    end: haystack.len(),
                    is_match_fw: true,
                    is_match_bw: true,
                    is_finished: false,
                }),
            }
        } else {
            StrSearcher {
                haystack,
                needle,
                searcher: StrSearcherImpl::TwoWay(TwoWaySearcher::new(
                    needle.as_bytes(),
                    haystack.len(),
                )),
            }
        }
    }
}

unsafe impl<'a, 'b> Searcher<'a> for StrSearcher<'a, 'b> {
    #[inline]
    fn haystack(&self) -> &'a str {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        match self.searcher {
            StrSearcherImpl::Empty(ref mut searcher) => {
                if searcher.is_finished {
                    return SearchStep::Done;
                }
                // empty needle rejects every char and matches every empty string between them
                let is_match = searcher.is_match_fw;
                searcher.is_match_fw = !searcher.is_match_fw;
                let pos = searcher.position;
                match self.haystack[pos..].chars().next() {
                    _ if is_match => SearchStep::Match(pos, pos),
                    None => {
                        searcher.is_finished = true;
                        SearchStep::Done
                    }
                    Some(ch) => {
                        searcher.position += ch.len_utf8();
                        SearchStep::Reject(pos, searcher.position)
                    }
                }
            }
            StrSearcherImpl::TwoWay(ref mut searcher) => {
                // TwoWaySearcher produces valid *Match* indices that split at char boundaries
                // as long as it does correct matching and that haystack and needle are
                // valid UTF-8
                // *Rejects* from the algorithm can fall on any indices, but we will walk them
                // manually to the next character boundary, so that they are utf-8 safe.
                if searcher.position == self.haystack.len() {
                    return SearchStep::Done;
                }
                let is_long = searcher.memory == usize::MAX;
                match searcher.next::<RejectAndMatch>(
                    self.haystack.as_bytes(),
                    self.needle.as_bytes(),
                    is_long,
                ) {
                    byte_pattern::SearchStep::Reject(a, mut b) => {
                        // skip to next char boundary
                        while !self.haystack.is_char_boundary(b) {
                            b += 1;
                        }
                        searcher.position = cmp::max(b, searcher.position);
                        SearchStep::Reject(a, b)
                    }
                    byte_pattern::SearchStep::Match(a, b) => SearchStep::Match(a, b),
                    byte_pattern::SearchStep::Done => SearchStep::Done,
                }
            }
        }
    }

    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        match self.searcher {
            StrSearcherImpl::Empty(..) => loop {
                match self.next() {
                    SearchStep::Match(a, b) => return Some((a, b)),
                    SearchStep::Done => return None,
                    SearchStep::Reject(..) => {}
                }
            },
            StrSearcherImpl::TwoWay(ref mut searcher) => {
                let is_long = searcher.memory == usize::MAX;
                // write out `true` and `false` cases to encourage the compiler
                // to specialize the two cases separately.
                if is_long {
                    searcher.next::<MatchOnly>(
                        self.haystack.as_bytes(),
                        self.needle.as_bytes(),
                        true,
                    )
                } else {
                    searcher.next::<MatchOnly>(
                        self.haystack.as_bytes(),
                        self.needle.as_bytes(),
                        false,
                    )
                }
            }
        }
    }
}

unsafe impl<'a, 'b> ReverseSearcher<'a> for StrSearcher<'a, 'b> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        match self.searcher {
            StrSearcherImpl::Empty(ref mut searcher) => {
                if searcher.is_finished {
                    return SearchStep::Done;
                }
                let is_match = searcher.is_match_bw;
                searcher.is_match_bw = !searcher.is_match_bw;
                let end = searcher.end;
                match self.haystack[..end].chars().next_back() {
                    _ if is_match => SearchStep::Match(end, end),
                    None => {
                        searcher.is_finished = true;
                        SearchStep::Done
                    }
                    Some(ch) => {
                        searcher.end -= ch.len_utf8();
                        SearchStep::Reject(searcher.end, end)
                    }
                }
            }
            StrSearcherImpl::TwoWay(ref mut searcher) => {
                if searcher.end == 0 {
                    return SearchStep::Done;
                }
                let is_long = searcher.memory == usize::MAX;
                match searcher.next_back::<RejectAndMatch>(
                    self.haystack.as_bytes(),
                    self.needle.as_bytes(),
                    is_long,
                ) {
                    byte_pattern::SearchStep::Reject(mut a, b) => {
                        // skip to next char boundary
                        while !self.haystack.is_char_boundary(a) {
                            a -= 1;
                        }
                        searcher.end = cmp::min(a, searcher.end);
                        SearchStep::Reject(a, b)
                    }
                    byte_pattern::SearchStep::Match(a, b) => SearchStep::Match(a, b),
                    byte_pattern::SearchStep::Done => SearchStep::Done,
                }
            }
        }
    }

    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        match self.searcher {
            StrSearcherImpl::Empty(..) => loop {
                match self.next_back() {
                    SearchStep::Match(a, b) => return Some((a, b)),
                    SearchStep::Done => return None,
                    SearchStep::Reject(..) => {}
                }
            },
            StrSearcherImpl::TwoWay(ref mut searcher) => {
                let is_long = searcher.memory == usize::MAX;
                // write out `true` and `false`, like `next_match`
                if is_long {
                    searcher.next_back::<MatchOnly>(
                        self.haystack.as_bytes(),
                        self.needle.as_bytes(),
                        true,
                    )
                } else {
                    searcher.next_back::<MatchOnly>(
                        self.haystack.as_bytes(),
                        self.needle.as_bytes(),
                        false,
                    )
                }
            }
        }
    }
}
