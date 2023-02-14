//! The Pattern API.
//!
//! The Pattern API provides a generic mechanism for using different pattern
//! types when searching through different objects.
//!
//! For more details, see the traits [`Pattern`], [`Haystack`], [`Searcher`],
//! [`ReverseSearcher`] and [`DoubleEndedSearcher`].  Although this API is
//! unstable, it is exposed via stable methods on corresponding haystack types.
//!
//! # Examples
//!
//! [`Pattern<&str>`] is [implemented][pattern-impls] in the stable API for
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

use crate::ops::Range;

/// A pattern which can be matched against a [`Haystack`].
///
/// A `Pattern<H>` expresses that the implementing type can be used as a pattern
/// for searching in an `H`.  For example, character `'a'` and string `"aa"` are
/// patterns that would match at index `1` in the string `"baaaab"`.
///
/// The trait itself acts as a builder for an associated [`Searcher`] type,
/// which does the actual work of finding occurrences of the pattern in
/// a string.
///
/// Depending on the type of the haystack and the pattern, the semantics of the
/// pattern can change.  The table below describes some of those behaviours for
/// a [`&str`][str] haystack.
///
/// | Pattern type             | Match condition                           |
/// |--------------------------|-------------------------------------------|
/// | `&str`                   | is substring                              |
/// | `char`                   | is contained in string                    |
/// | `&[char]`                | any char in slice is contained in string  |
/// | `F: FnMut(char) -> bool` | `F` returns `true` for a char in string   |
///
/// # Examples
///
/// ```
/// // &str pattern matching &str
/// assert_eq!("abaaa".find("ba"), Some(1));
/// assert_eq!("abaaa".find("bac"), None);
///
/// // char pattern matching &str
/// assert_eq!("abaaa".find('a'), Some(0));
/// assert_eq!("abaaa".find('b'), Some(1));
/// assert_eq!("abaaa".find('c'), None);
///
/// // &[char; N] pattern matching &str
/// assert_eq!("ab".find(&['b', 'a']), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z']), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd']), None);
///
/// // &[char] pattern matching &str
/// assert_eq!("ab".find(&['b', 'a'][..]), Some(0));
/// assert_eq!("abaaa".find(&['a', 'z'][..]), Some(0));
/// assert_eq!("abaaa".find(&['c', 'd'][..]), None);
///
/// // FnMut(char) -> bool pattern matching &str
/// assert_eq!("abcdef_z".find(|ch| ch > 'd' && ch < 'y'), Some(4));
/// assert_eq!("abcddd_z".find(|ch| ch > 'd' && ch < 'y'), None);
/// ```
pub trait Pattern<H: Haystack>: Sized {
    /// Associated searcher for this pattern.
    type Searcher: Searcher<H>;

    /// Constructs the associated searcher from `self` and the `haystack` to
    /// search in.
    fn into_searcher(self, haystack: H) -> Self::Searcher;

    /// Checks whether the pattern matches anywhere in the haystack.
    fn is_contained_in(self, haystack: H) -> bool {
        self.into_searcher(haystack).next_match().is_some()
    }

    /// Checks whether the pattern matches at the front of the haystack.
    fn is_prefix_of(self, haystack: H) -> bool {
        matches!(self.into_searcher(haystack).next(), SearchStep::Match(..))
    }

    /// Checks whether the pattern matches at the back of the haystack.
    fn is_suffix_of(self, haystack: H) -> bool
    where
        Self::Searcher: ReverseSearcher<H>,
    {
        matches!(self.into_searcher(haystack).next_back(), SearchStep::Match(..))
    }

    /// Removes the pattern from the front of haystack, if it matches.
    fn strip_prefix_of(self, haystack: H) -> Option<H> {
        if let SearchStep::Match(start, pos) = self.into_searcher(haystack).next() {
            // This cannot be debug_assert_eq because StartCursor isn’t Debug.
            debug_assert!(
                start == haystack.cursor_at_front(),
                "The first search step from Searcher \
                 must include the first character"
            );
            let end = haystack.cursor_at_back();
            // SAFETY: `Searcher` is known to return valid indices.
            Some(unsafe { haystack.get_unchecked(pos..end) })
        } else {
            None
        }
    }

    /// Removes the pattern from the back of haystack, if it matches.
    fn strip_suffix_of(self, haystack: H) -> Option<H>
    where
        Self::Searcher: ReverseSearcher<H>,
    {
        if let SearchStep::Match(pos, end) = self.into_searcher(haystack).next_back() {
            // This cannot be debug_assert_eq because StartCursor isn’t Debug.
            debug_assert!(
                end == haystack.cursor_at_back(),
                "The first search step from ReverseSearcher \
                 must include the last character"
            );
            let start = haystack.cursor_at_front();
            // SAFETY: `Searcher` is known to return valid indices.
            Some(unsafe { haystack.get_unchecked(start..pos) })
        } else {
            None
        }
    }
}

/// A type which can be searched in using a [`Pattern`].
///
/// The trait is used in combination with [`Pattern`] trait to express a pattern
/// that can be used to search for elements in given haystack.
pub trait Haystack: Sized + Copy {
    /// A cursor representing position in the haystack or its end.
    type Cursor: Copy + PartialEq;

    /// Returns cursor pointing at the beginning of the haystack.
    fn cursor_at_front(self) -> Self::Cursor;

    /// Returns cursor pointing at the end of the haystack.
    fn cursor_at_back(self) -> Self::Cursor;

    /// Returns whether the haystack is empty.
    fn is_empty(self) -> bool;

    /// Returns portions of the haystack indicated by the cursor range.
    ///
    /// # Safety
    ///
    /// Range’s start and end must be valid haystack split positions.
    /// Furthermore, start mustn’t point at position after end.
    ///
    /// A valid split positions are:
    /// - the front of the haystack (as returned by
    ///   [`cursor_at_front()`][Self::cursor_at_front],
    /// - the back of the haystack (as returned by
    ///   [`cursor_at_back()`][Self::cursor_at_back] or
    /// - any cursor returned by a [`Searcher`] or [`ReverseSearcher`].
    unsafe fn get_unchecked(self, range: Range<Self::Cursor>) -> Self;
}

/// Result of calling [`Searcher::next()`] or [`ReverseSearcher::next_back()`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SearchStep<T = usize> {
    /// Expresses that a match of the pattern has been found at
    /// `haystack[a..b]`.
    Match(T, T),
    /// Expresses that `haystack[a..b]` has been rejected as a possible match of
    /// the pattern.
    ///
    /// Note that there might be more than one `Reject` between two `Match`es,
    /// there is no requirement for them to be combined into one.
    Reject(T, T),
    /// Expresses that every element of the haystack has been visited, ending
    /// the iteration.
    Done,
}

/// A searcher for a string pattern.
///
/// This trait provides methods for searching for non-overlapping matches of
/// a pattern starting from the front of a haystack `H`.
///
/// It will be implemented by associated `Searcher` types of the [`Pattern`]
/// trait.
///
/// The trait is marked unsafe because the indices returned by the
/// [`next()`][Searcher::next] methods are required to lie on valid haystack
/// split positions.  This enables consumers of this trait to slice the haystack
/// without additional runtime checks.
pub unsafe trait Searcher<H: Haystack> {
    /// Getter for the underlying string to be searched in
    ///
    /// Will always return the same haystack that was used when creating the
    /// searcher.
    fn haystack(&self) -> H;

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
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"` might
    /// produce the stream `[Reject(0, 1), Reject(1, 2), Match(2, 5), Reject(5,
    /// 8)]`
    fn next(&mut self) -> SearchStep<H::Cursor>;

    /// Finds the next [`Match`][SearchStep::Match] result. See
    /// [`next()`][Searcher::next].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the
    /// returned ranges of this and [`next_reject`][Searcher::next_reject] will
    /// overlap.  This will return `(start_match, end_match)`, where start_match
    /// is the index of where the match begins, and end_match is the index after
    /// the end of the match.
    fn next_match(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop {
            match self.next() {
                SearchStep::Match(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }

    /// Finds the next [`Reject`][SearchStep::Reject] result.  See
    /// [`next()`][Searcher::next] and [`next_match()`][Searcher::next_match].
    ///
    /// Unlike [`next()`][Searcher::next], there is no guarantee that the
    /// returned ranges of this and [`next_match`][Searcher::next_match] will
    /// overlap.
    fn next_reject(&mut self) -> Option<(H::Cursor, H::Cursor)> {
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
/// This trait provides methods for searching for non-overlapping matches of
/// a pattern starting from the back of a haystack `H`.
///
/// It will be implemented by associated [`Searcher`] types of the [`Pattern`]
/// trait if the pattern supports searching for it from the back.
///
/// The index ranges returned by this trait are not required to exactly match
/// those of the forward search in reverse.
///
/// For the reason why this trait is marked unsafe, see the parent trait
/// [`Searcher`].
pub unsafe trait ReverseSearcher<H: Haystack>: Searcher<H> {
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
    /// will contain index ranges that are adjacent, non-overlapping, covering
    /// the whole haystack, and laying on utf8 boundaries.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero
    /// length.
    ///
    /// As an example, the pattern `"aaa"` and the haystack `"cbaaaaab"` might
    /// produce the stream `[Reject(7, 8), Match(4, 7), Reject(1, 4), Reject(0,
    /// 1)]`.
    fn next_back(&mut self) -> SearchStep<H::Cursor>;

    /// Finds the next [`Match`][SearchStep::Match] result.
    /// See [`next_back()`][ReverseSearcher::next_back].
    fn next_match_back(&mut self) -> Option<(H::Cursor, H::Cursor)> {
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
    fn next_reject_back(&mut self) -> Option<(H::Cursor, H::Cursor)> {
        loop {
            match self.next_back() {
                SearchStep::Reject(a, b) => return Some((a, b)),
                SearchStep::Done => return None,
                _ => continue,
            }
        }
    }
}

/// A marker trait to express that a [`ReverseSearcher`] can be used for
/// a [`DoubleEndedIterator`] implementation.
///
/// For this, the impl of [`Searcher`] and [`ReverseSearcher`] need to follow
/// these conditions:
///
/// - All results of `next()` need to be identical to the results of
///   `next_back()` in reverse order.
/// - `next()` and `next_back()` need to behave as the two ends of a range of
///   values, that is they can not "walk past each other".
///
/// # Examples
///
/// `char::Searcher` is a `DoubleEndedSearcher` because searching for a [`char`]
/// only requires looking at one at a time, which behaves the same from both
/// ends.
///
/// `(&str)::Searcher` is not a `DoubleEndedSearcher` because the pattern `"aa"`
/// in the haystack `"aaa"` matches as either `"[aa]a"` or `"a[aa]"`, depending
/// from which side it is searched.
pub trait DoubleEndedSearcher<H: Haystack>: ReverseSearcher<H> {}
