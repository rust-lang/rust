//! The BytePattern API.
//!
//! The BytePattern API provides a generic mechanism for using different pattern
//! types when searching through a byte slice.
//!
//! For more details, see the traits [`BytePattern`], [`ByteSearcher`],
//! [`ReverseByteSearcher`].
//!
//! Although this API is unstable, it is exposed via stable APIs on the
//! [`&[u8]`](prim@slice) type.
//!
//! # Examples
//!
//! [`BytePattern`] is [implemented][pattern-impls] in the stable API for
//! [`&[u8]`](prim@slice), [`u8`], and functions and closures implementing `FnMut(u8) -> bool`.
//!
//! ```
//! #![feature(byte_search)]
//! let s = b"Can you find a needle in a haystack?";
//!
//! // &[u8] pattern
//! assert_eq!(s.find_bytes(b"you"), Some(4));
//! // u8 pattern
//! assert_eq!(s.find_bytes(b'n'), Some(2));
//! // closure pattern
//! assert_eq!(s.find_bytes(|b: u8| b == b'?'), Some(35));
//! ```
//!
//! [pattern-impls]: BytePattern#implementors

#![unstable(
    feature = "byte_search",
    reason = "API not fully fleshed out and ready to be stabilized",
    issue = "134149"
)]

use crate::cmp::Ordering;
use crate::{cmp, fmt};

// Pattern

/// A byte pattern.
///
/// A `BytePattern` expresses that the implementing type
/// can be used as a byte pattern for searching in a [`&[u8]`][slice].
///
/// For example, both `20u8` and `&[20u8, 20]` are patterns that
/// would match at index `1` in the slice `&[19u8, 20, 20, 21]`.
///
/// The trait itself acts as a builder for an associated
/// [`ByteSearcher`] type, which does the actual work of finding
/// occurrences of the pattern in a string.
///
/// Depending on the type of the pattern, the behavior of methods like
/// [`slice::find_bytes`] and [`slice::contains_bytes`] can change. The table below describes
/// some of those behaviors.
///
/// | Pattern type             | Match condition                           |
/// |--------------------------|-------------------------------------------|
/// | `&[u8]`                  | is subslice                               |
/// | `u8`                     | is contained in slice                     |
/// | `F: FnMut(u8) -> bool`   | `F` returns `true` for a u8 in slice      |
/// | `&&[u8]`                 | is subslice                               |
/// | `&Vec<u8>`               | is subslice                               |
///
/// # Examples
///
/// ```
/// #![feature(byte_search)]
///
/// // &[u8]
/// assert_eq!(b"abaaa".find_bytes(b"ba"), Some(1));
/// assert_eq!(b"abaaa".find_bytes(b"bac"), None);
///
/// // u8
/// assert_eq!(b"abaaa".find_bytes(b'a'), Some(0));
/// assert_eq!(b"abaaa".find_bytes(b'b'), Some(1));
/// assert_eq!(b"abaaa".find_bytes(b'c'), None);
///
/// // FnMut(u8) -> bool
/// assert_eq!(b"abcdef_z".find_bytes(|ch| ch > b'd' && ch < b'y'), Some(4));
/// assert_eq!(b"abcddd_z".find_bytes(|ch| ch > b'd' && ch < b'y'), None);
/// ```
pub trait BytePattern: Sized {
    /// Associated searcher for this pattern
    type ByteSearcher<'a>: ByteSearcher<'a>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: &[u8]) -> Self::ByteSearcher<'_>;

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &[u8]) -> bool {
        self.into_searcher(haystack).next_match().is_some()
    }

    /// Checks whether the pattern matches at the front of the haystack
    #[inline]
    fn is_prefix_of(self, haystack: &[u8]) -> bool {
        matches!(self.into_searcher(haystack).next(), SearchStep::Match(0, _))
    }

    /// Checks whether the pattern matches at the back of the haystack
    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a [u8]) -> bool
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        matches!(self.into_searcher(haystack).next_back(), SearchStep::Match(_, j) if haystack.len() == j)
    }

    /// Removes the pattern from the front of haystack, if it matches.
    #[inline]
    fn strip_prefix_of(self, haystack: &[u8]) -> Option<&[u8]> {
        if let SearchStep::Match(start, len) = self.into_searcher(haystack).next() {
            debug_assert_eq!(
                start, 0,
                "The first search step from ByteSearcher \
                 must include the first byte"
            );
            // SAFETY: `ByteSearcher` is known to return valid indices.
            unsafe { Some(haystack.get_unchecked(len..)) }
        } else {
            None
        }
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        if let SearchStep::Match(start, end) = self.into_searcher(haystack).next_back() {
            debug_assert_eq!(
                end,
                haystack.len(),
                "The first search step from ReverseByteSearcher \
                 must include the last byte"
            );
            // SAFETY: `ByteSearcher` is known to return valid indices.
            unsafe { Some(haystack.get_unchecked(..start)) }
        } else {
            None
        }
    }
}

// ByteSearcher

/// Result of calling [`ByteSearcher::next()`] or [`ReverseByteSearcher::next_back()`].
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
/// matches of a pattern starting from the front (left) of a byte slice.
///
/// It will be implemented by associated `ByteSearcher`
/// types of the [`BytePattern`] trait.
///
/// The trait is marked unsafe because the indices returned by the
/// [`next()`][ByteSearcher::next] methods are required to be in-bounds
/// in the haystack. This enables consumers of this trait to
/// slice the haystack without additional runtime checks.
pub unsafe trait ByteSearcher<'a> {
    /// Getter for the underlying string to be searched in
    ///
    /// Will always return the same [`&[u8]`][slice].
    fn haystack(&self) -> &'a [u8];

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
    /// covering the whole haystack.
    ///
    /// A [`Match`][SearchStep::Match] result needs to contain the whole matched
    /// pattern, however [`Reject`][SearchStep::Reject] results may be split up
    /// into arbitrary many adjacent fragments. Both ranges may have zero length.
    ///
    /// As an example, the pattern `b"aaa"` and the haystack `b"cbaaaaab"`
    /// might produce the stream
    /// `[Reject(0, 1), Reject(1, 2), Match(2, 5), Reject(5, 8)]`
    fn next(&mut self) -> SearchStep;

    /// Finds the next [`Match`][SearchStep::Match] result. See [`next()`][ByteSearcher::next].
    ///
    /// Unlike [`next()`][ByteSearcher::next], there is no guarantee that the returned ranges
    /// of this and [`next_reject`][ByteSearcher::next_reject] will overlap. This will return
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

    /// Finds the next [`Reject`][SearchStep::Reject] result. See [`next()`][ByteSearcher::next]
    /// and [`next_match()`][ByteSearcher::next_match].
    ///
    /// Unlike [`next()`][ByteSearcher::next], there is no guarantee that the returned ranges
    /// of this and [`next_match`][ByteSearcher::next_match] will overlap.
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

/// A reverse searcher for a byte pattern.
///
/// This trait provides methods for searching for non-overlapping
/// matches of a pattern starting from the back (right) of a byte slice.
///
/// It will be implemented by associated [`ByteSearcher`]
/// types of the [`BytePattern`] trait if the pattern supports searching
/// for it from the back.
///
/// The index ranges returned by this trait are not required
/// to exactly match those of the forward search in reverse.
///
/// For the reason why this trait is marked unsafe, see the
/// parent trait [`ByteSearcher`].
pub unsafe trait ReverseByteSearcher<'a>: ByteSearcher<'a> {
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
    /// covering the whole haystack.
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
    /// See [`next_back()`][ReverseByteSearcher::next_back].
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
    /// See [`next_back()`][ReverseByteSearcher::next_back].
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

/////////////////////////////////////////////////////////////////////////////
// Impl for F: FnMut(u8) -> bool
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<F as Pattern>::ByteSearcher<'a>`.
#[derive(Clone)]
pub struct BytePredicateByteSearcher<'a, F: FnMut(u8) -> bool> {
    predicate: F,
    haystack: &'a [u8],
    iter: core::iter::Enumerate<core::slice::Iter<'a, u8>>,
}

impl<F> fmt::Debug for BytePredicateByteSearcher<'_, F>
where
    F: FnMut(u8) -> bool,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytePredicateByteSearcher")
            .field("haystack", &self.haystack)
            .field("iter", &self.iter)
            .finish()
    }
}

unsafe impl<'a, F> ByteSearcher<'a> for BytePredicateByteSearcher<'a, F>
where
    F: FnMut(u8) -> bool,
{
    fn haystack(&self) -> &'a [u8] {
        self.haystack
    }

    fn next(&mut self) -> SearchStep {
        match self.iter.next() {
            None => SearchStep::Done,
            Some((index, byte)) => {
                if (self.predicate)(*byte) {
                    SearchStep::Match(index, index + 1)
                } else {
                    SearchStep::Reject(index, index + 1)
                }
            }
        }
    }
}

unsafe impl<'a, F> ReverseByteSearcher<'a> for BytePredicateByteSearcher<'a, F>
where
    F: FnMut(u8) -> bool,
{
    fn next_back(&mut self) -> SearchStep {
        match self.iter.next_back() {
            None => SearchStep::Done,
            Some((index, byte)) => {
                if (self.predicate)(*byte) {
                    SearchStep::Match(index, index + 1)
                } else {
                    SearchStep::Reject(index, index + 1)
                }
            }
        }
    }
}

/// Searches for [`u8`]s that match the given predicate.
///
/// # Examples
///
/// ```
/// #![feature(byte_search)]
///
/// assert_eq!(b"Hello world".find_bytes(|b| b != b' '), Some(0));
/// assert_eq!(b"Hello world".find_bytes(|b| b"aeiou".contains(&b)), Some(1));
/// ```
impl<F> BytePattern for F
where
    F: FnMut(u8) -> bool,
{
    /// Associated searcher for this pattern
    type ByteSearcher<'a> = BytePredicateByteSearcher<'a, F>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: &[u8]) -> Self::ByteSearcher<'_> {
        BytePredicateByteSearcher { predicate: self, haystack, iter: haystack.iter().enumerate() }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for u8
/////////////////////////////////////////////////////////////////////////////

/// Associated type for `<u8 as Pattern>::ByteSearcher<'a>`.
#[derive(Debug)]
pub struct ByteValueByteSearcher<'a> {
    value: u8,
    haystack: &'a [u8],
    iter: core::iter::Enumerate<core::slice::Iter<'a, u8>>,
}

unsafe impl<'a> ByteSearcher<'a> for ByteValueByteSearcher<'a> {
    fn haystack(&self) -> &'a [u8] {
        self.haystack
    }

    fn next(&mut self) -> SearchStep {
        match self.iter.next() {
            None => SearchStep::Done,
            Some((index, byte)) => {
                if *byte == self.value {
                    SearchStep::Match(index, index + 1)
                } else {
                    SearchStep::Reject(index, index + 1)
                }
            }
        }
    }
}

unsafe impl<'a> ReverseByteSearcher<'a> for ByteValueByteSearcher<'a> {
    fn next_back(&mut self) -> SearchStep {
        match self.iter.next_back() {
            None => SearchStep::Done,
            Some((index, byte)) => {
                if *byte == self.value {
                    SearchStep::Match(index, index + 1)
                } else {
                    SearchStep::Reject(index, index + 1)
                }
            }
        }
    }
}

///
impl BytePattern for u8 {
    /// Associated searcher for this pattern
    type ByteSearcher<'a> = ByteValueByteSearcher<'a>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: &[u8]) -> Self::ByteSearcher<'_> {
        ByteValueByteSearcher { value: self, haystack, iter: haystack.iter().enumerate() }
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &[u8]) -> bool {
        haystack.contains(&self)
    }

    #[inline]
    fn is_prefix_of(self, haystack: &[u8]) -> bool {
        haystack.first().copied() == Some(self)
    }

    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a [u8]) -> bool
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        haystack.last().copied() == Some(self)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &[u8]) -> Option<&[u8]> {
        match haystack {
            [first, rest @ ..] if *first == self => Some(rest),
            _ => None,
        }
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        match haystack {
            [rest @ .., last] if *last == self => Some(rest),
            _ => None,
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &[u8; N]
/////////////////////////////////////////////////////////////////////////////

/// Delegates to the `&[u8]` impl.
impl<'b, const N: usize> BytePattern for &'b [u8; N] {
    /// Associated searcher for this pattern
    type ByteSearcher<'a> = ByteSliceByteSearcher<'a, 'b>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: &[u8]) -> Self::ByteSearcher<'_> {
        self.as_slice().into_searcher(haystack)
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &[u8]) -> bool {
        self.as_slice().is_contained_in(haystack)
    }

    #[inline]
    fn is_prefix_of(self, haystack: &[u8]) -> bool {
        self.as_slice().is_prefix_of(haystack)
    }

    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a [u8]) -> bool
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        self.as_slice().is_suffix_of(haystack)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &[u8]) -> Option<&[u8]> {
        self.as_slice().strip_prefix_of(haystack)
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        self.as_slice().strip_suffix_of(haystack)
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &&[u8]
/////////////////////////////////////////////////////////////////////////////

/// Delegates to the `&[u8]` impl.
impl<'b, 'c> BytePattern for &'c &'b [u8] {
    /// Associated searcher for this pattern
    type ByteSearcher<'a> = ByteSliceByteSearcher<'a, 'b>;

    /// Constructs the associated searcher from
    /// `self` and the `haystack` to search in.
    fn into_searcher(self, haystack: &[u8]) -> Self::ByteSearcher<'_> {
        (*self).into_searcher(haystack)
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &[u8]) -> bool {
        (*self).is_contained_in(haystack)
    }

    #[inline]
    fn is_prefix_of(self, haystack: &[u8]) -> bool {
        (*self).is_prefix_of(haystack)
    }

    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a [u8]) -> bool
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        (*self).is_suffix_of(haystack)
    }

    #[inline]
    fn strip_prefix_of(self, haystack: &[u8]) -> Option<&[u8]> {
        (*self).strip_prefix_of(haystack)
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        (*self).strip_suffix_of(haystack)
    }
}

/////////////////////////////////////////////////////////////////////////////
// Impl for &[u8]
/////////////////////////////////////////////////////////////////////////////

/// Non-allocating subslice search.
///
/// Will handle the pattern `&[]` as returning empty matches at each index.
///
/// # Examples
///
/// ```
/// #![feature(byte_search)]
///
/// assert_eq!(b"Hello world".find_bytes(b"world"), Some(6));
/// ```
impl<'b> BytePattern for &'b [u8] {
    type ByteSearcher<'a> = ByteSliceByteSearcher<'a, 'b>;

    #[inline]
    fn into_searcher(self, haystack: &[u8]) -> ByteSliceByteSearcher<'_, 'b> {
        ByteSliceByteSearcher::new(haystack, self)
    }

    /// Checks whether the pattern matches at the front of the haystack.
    #[inline]
    fn is_prefix_of(self, haystack: &[u8]) -> bool {
        haystack.starts_with(self)
    }

    /// Checks whether the pattern matches anywhere in the haystack
    #[inline]
    fn is_contained_in(self, haystack: &[u8]) -> bool {
        if self.len() == 0 {
            return true;
        }

        match self.len().cmp(&haystack.len()) {
            Ordering::Less => {
                if self.len() == 1 {
                    return haystack.contains(&self[0]);
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
    fn strip_prefix_of(self, haystack: &[u8]) -> Option<&[u8]> {
        if self.is_prefix_of(haystack) {
            // SAFETY: prefix was just verified to exist.
            unsafe { Some(haystack.get_unchecked(self.len()..)) }
        } else {
            None
        }
    }

    /// Checks whether the pattern matches at the back of the haystack.
    #[inline]
    fn is_suffix_of<'a>(self, haystack: &'a [u8]) -> bool
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        haystack.ends_with(self)
    }

    /// Removes the pattern from the back of haystack, if it matches.
    #[inline]
    fn strip_suffix_of<'a>(self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        Self::ByteSearcher<'a>: ReverseByteSearcher<'a>,
    {
        if self.is_suffix_of(haystack) {
            let i = haystack.len() - self.len();
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
/// Associated type for `<&[u8] as Pattern>::ByteSearcher<'a>`.
pub struct ByteSliceByteSearcher<'a, 'b> {
    haystack: &'a [u8],
    needle: &'b [u8],

    searcher: ByteSliceByteSearcherImpl,
}

#[derive(Clone, Debug)]
enum ByteSliceByteSearcherImpl {
    Empty(EmptyNeedle),
    TwoWay(TwoWayByteSearcher),
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

impl<'a, 'b> ByteSliceByteSearcher<'a, 'b> {
    fn new(haystack: &'a [u8], needle: &'b [u8]) -> ByteSliceByteSearcher<'a, 'b> {
        if needle.is_empty() {
            ByteSliceByteSearcher {
                haystack,
                needle,
                searcher: ByteSliceByteSearcherImpl::Empty(EmptyNeedle {
                    position: 0,
                    end: haystack.len(),
                    is_match_fw: true,
                    is_match_bw: true,
                    is_finished: false,
                }),
            }
        } else {
            ByteSliceByteSearcher {
                haystack,
                needle,
                searcher: ByteSliceByteSearcherImpl::TwoWay(TwoWayByteSearcher::new(
                    needle,
                    haystack.len(),
                )),
            }
        }
    }
}

unsafe impl<'a, 'b> ByteSearcher<'a> for ByteSliceByteSearcher<'a, 'b> {
    #[inline]
    fn haystack(&self) -> &'a [u8] {
        self.haystack
    }

    #[inline]
    fn next(&mut self) -> SearchStep {
        match self.searcher {
            ByteSliceByteSearcherImpl::Empty(ref mut searcher) => {
                if searcher.is_finished {
                    return SearchStep::Done;
                }
                // empty needle rejects every byte and matches every empty slice between them
                let is_match = searcher.is_match_fw;
                searcher.is_match_fw = !searcher.is_match_fw;
                let pos = searcher.position;
                match self.haystack[pos..].iter().next() {
                    _ if is_match => SearchStep::Match(pos, pos),
                    None => {
                        searcher.is_finished = true;
                        SearchStep::Done
                    }
                    Some(_) => {
                        searcher.position += 1;
                        SearchStep::Reject(pos, searcher.position)
                    }
                }
            }
            ByteSliceByteSearcherImpl::TwoWay(ref mut searcher) => {
                if searcher.position == self.haystack.len() {
                    return SearchStep::Done;
                }
                let is_long = searcher.memory == usize::MAX;
                searcher.next::<RejectAndMatch>(self.haystack, self.needle, is_long)
            }
        }
    }

    #[inline]
    fn next_match(&mut self) -> Option<(usize, usize)> {
        match self.searcher {
            ByteSliceByteSearcherImpl::Empty(..) => loop {
                match self.next() {
                    SearchStep::Match(a, b) => return Some((a, b)),
                    SearchStep::Done => return None,
                    SearchStep::Reject(..) => {}
                }
            },
            ByteSliceByteSearcherImpl::TwoWay(ref mut searcher) => {
                let is_long = searcher.memory == usize::MAX;
                // write out `true` and `false` cases to encourage the compiler
                // to specialize the two cases separately.
                if is_long {
                    searcher.next::<MatchOnly>(self.haystack, self.needle, true)
                } else {
                    searcher.next::<MatchOnly>(self.haystack, self.needle, false)
                }
            }
        }
    }
}

unsafe impl<'a, 'b> ReverseByteSearcher<'a> for ByteSliceByteSearcher<'a, 'b> {
    #[inline]
    fn next_back(&mut self) -> SearchStep {
        match self.searcher {
            ByteSliceByteSearcherImpl::Empty(ref mut searcher) => {
                if searcher.is_finished {
                    return SearchStep::Done;
                }
                let is_match = searcher.is_match_bw;
                searcher.is_match_bw = !searcher.is_match_bw;
                let end = searcher.end;
                match self.haystack[..end].iter().next_back() {
                    _ if is_match => SearchStep::Match(end, end),
                    None => {
                        searcher.is_finished = true;
                        SearchStep::Done
                    }
                    Some(_) => {
                        searcher.end -= 1;
                        SearchStep::Reject(searcher.end, end)
                    }
                }
            }
            ByteSliceByteSearcherImpl::TwoWay(ref mut searcher) => {
                if searcher.end == 0 {
                    return SearchStep::Done;
                }
                let is_long = searcher.memory == usize::MAX;
                searcher.next_back::<RejectAndMatch>(self.haystack, self.needle, is_long)
            }
        }
    }

    #[inline]
    fn next_match_back(&mut self) -> Option<(usize, usize)> {
        match self.searcher {
            ByteSliceByteSearcherImpl::Empty(..) => loop {
                match self.next_back() {
                    SearchStep::Match(a, b) => return Some((a, b)),
                    SearchStep::Done => return None,
                    SearchStep::Reject(..) => {}
                }
            },
            ByteSliceByteSearcherImpl::TwoWay(ref mut searcher) => {
                let is_long = searcher.memory == usize::MAX;
                // write out `true` and `false`, like `next_match`
                if is_long {
                    searcher.next_back::<MatchOnly>(self.haystack, self.needle, true)
                } else {
                    searcher.next_back::<MatchOnly>(self.haystack, self.needle, false)
                }
            }
        }
    }
}

/// The internal state of the two-way substring search algorithm.
#[derive(Clone, Debug)]
struct TwoWayByteSearcher {
    // constants
    /// critical factorization index
    crit_pos: usize,
    /// critical factorization index for reversed needle
    crit_pos_back: usize,
    period: usize,
    /// `byteset` is an extension (not part of the two way algorithm);
    /// it's a 64-bit "fingerprint" where each set bit `j` corresponds
    /// to a (byte & 63) == j present in the needle.
    byteset: u64,

    // variables
    position: usize,
    end: usize,
    /// index into needle before which we have already matched
    memory: usize,
    /// index into needle after which we have already matched
    memory_back: usize,
}

/*
    This is the Two-Way search algorithm, which was introduced in the paper:
    Crochemore, M., Perrin, D., 1991, Two-way string-matching, Journal of the ACM 38(3):651-675.

    Here's some background information.

    A *word* is a string of symbols. The *length* of a word should be a familiar
    notion, and here we denote it for any word x by |x|.
    (We also allow for the possibility of the *empty word*, a word of length zero).

    If x is any non-empty word, then an integer p with 0 < p <= |x| is said to be a
    *period* for x iff for all i with 0 <= i <= |x| - p - 1, we have x[i] == x[i+p].
    For example, both 1 and 2 are periods for the string "aa". As another example,
    the only period of the string "abcd" is 4.

    We denote by period(x) the *smallest* period of x (provided that x is non-empty).
    This is always well-defined since every non-empty word x has at least one period,
    |x|. We sometimes call this *the period* of x.

    If u, v and x are words such that x = uv, where uv is the concatenation of u and
    v, then we say that (u, v) is a *factorization* of x.

    Let (u, v) be a factorization for a word x. Then if w is a non-empty word such
    that both of the following hold

      - either w is a suffix of u or u is a suffix of w
      - either w is a prefix of v or v is a prefix of w

    then w is said to be a *repetition* for the factorization (u, v).

    Just to unpack this, there are four possibilities here. Let w = "abc". Then we
    might have:

      - w is a suffix of u and w is a prefix of v. ex: ("lolabc", "abcde")
      - w is a suffix of u and v is a prefix of w. ex: ("lolabc", "ab")
      - u is a suffix of w and w is a prefix of v. ex: ("bc", "abchi")
      - u is a suffix of w and v is a prefix of w. ex: ("bc", "a")

    Note that the word vu is a repetition for any factorization (u,v) of x = uv,
    so every factorization has at least one repetition.

    If x is a string and (u, v) is a factorization for x, then a *local period* for
    (u, v) is an integer r such that there is some word w such that |w| = r and w is
    a repetition for (u, v).

    We denote by local_period(u, v) the smallest local period of (u, v). We sometimes
    call this *the local period* of (u, v). Provided that x = uv is non-empty, this
    is well-defined (because each non-empty word has at least one factorization, as
    noted above).

    It can be proven that the following is an equivalent definition of a local period
    for a factorization (u, v): any positive integer r such that x[i] == x[i+r] for
    all i such that |u| - r <= i <= |u| - 1 and such that both x[i] and x[i+r] are
    defined. (i.e., i > 0 and i + r < |x|).

    Using the above reformulation, it is easy to prove that

        1 <= local_period(u, v) <= period(uv)

    A factorization (u, v) of x such that local_period(u,v) = period(x) is called a
    *critical factorization*.

    The algorithm hinges on the following theorem, which is stated without proof:

    **Critical Factorization Theorem** Any word x has at least one critical
    factorization (u, v) such that |u| < period(x).

    The purpose of maximal_suffix is to find such a critical factorization.

    If the period is short, compute another factorization x = u' v' to use
    for reverse search, chosen instead so that |v'| < period(x).

*/
impl TwoWayByteSearcher {
    fn new(needle: &[u8], end: usize) -> TwoWayByteSearcher {
        let (crit_pos_false, period_false) = TwoWayByteSearcher::maximal_suffix(needle, false);
        let (crit_pos_true, period_true) = TwoWayByteSearcher::maximal_suffix(needle, true);

        let (crit_pos, period) = if crit_pos_false > crit_pos_true {
            (crit_pos_false, period_false)
        } else {
            (crit_pos_true, period_true)
        };

        // A particularly readable explanation of what's going on here can be found
        // in Crochemore and Rytter's book "Text Algorithms", ch 13. Specifically
        // see the code for "Algorithm CP" on p. 323.
        //
        // What's going on is we have some critical factorization (u, v) of the
        // needle, and we want to determine whether u is a suffix of
        // &v[..period]. If it is, we use "Algorithm CP1". Otherwise we use
        // "Algorithm CP2", which is optimized for when the period of the needle
        // is large.
        if needle[..crit_pos] == needle[period..period + crit_pos] {
            // short period case -- the period is exact
            // compute a separate critical factorization for the reversed needle
            // x = u' v' where |v'| < period(x).
            //
            // This is sped up by the period being known already.
            // Note that a case like x = "acba" may be factored exactly forwards
            // (crit_pos = 1, period = 3) while being factored with approximate
            // period in reverse (crit_pos = 2, period = 2). We use the given
            // reverse factorization but keep the exact period.
            let crit_pos_back = needle.len()
                - cmp::max(
                    TwoWayByteSearcher::reverse_maximal_suffix(needle, period, false),
                    TwoWayByteSearcher::reverse_maximal_suffix(needle, period, true),
                );

            TwoWayByteSearcher {
                crit_pos,
                crit_pos_back,
                period,
                byteset: Self::byteset_create(&needle[..period]),

                position: 0,
                end,
                memory: 0,
                memory_back: needle.len(),
            }
        } else {
            // long period case -- we have an approximation to the actual period,
            // and don't use memorization.
            //
            // Approximate the period by lower bound max(|u|, |v|) + 1.
            // The critical factorization is efficient to use for both forward and
            // reverse search.

            TwoWayByteSearcher {
                crit_pos,
                crit_pos_back: crit_pos,
                period: cmp::max(crit_pos, needle.len() - crit_pos) + 1,
                byteset: Self::byteset_create(needle),

                position: 0,
                end,
                memory: usize::MAX, // Dummy value to signify that the period is long
                memory_back: usize::MAX,
            }
        }
    }

    #[inline]
    fn byteset_create(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0, |a, &b| (1 << (b & 0x3f)) | a)
    }

    #[inline]
    fn byteset_contains(&self, byte: u8) -> bool {
        (self.byteset >> ((byte & 0x3f) as usize)) & 1 != 0
    }

    // One of the main ideas of Two-Way is that we factorize the needle into
    // two halves, (u, v), and begin trying to find v in the haystack by scanning
    // left to right. If v matches, we try to match u by scanning right to left.
    // How far we can jump when we encounter a mismatch is all based on the fact
    // that (u, v) is a critical factorization for the needle.
    #[inline]
    fn next<S>(&mut self, haystack: &[u8], needle: &[u8], long_period: bool) -> S::Output
    where
        S: TwoWayStrategy,
    {
        // `next()` uses `self.position` as its cursor
        let old_pos = self.position;
        let needle_last = needle.len() - 1;
        'search: loop {
            // Check that we have room to search in
            // position + needle_last can not overflow if we assume slices
            // are bounded by isize's range.
            let tail_byte = match haystack.get(self.position + needle_last) {
                Some(&b) => b,
                None => {
                    self.position = haystack.len();
                    return S::rejecting(old_pos, self.position);
                }
            };

            if S::use_early_reject() && old_pos != self.position {
                return S::rejecting(old_pos, self.position);
            }

            // Quickly skip by large portions unrelated to our substring
            if !self.byteset_contains(tail_byte) {
                self.position += needle.len();
                if !long_period {
                    self.memory = 0;
                }
                continue 'search;
            }

            // See if the right part of the needle matches
            let start =
                if long_period { self.crit_pos } else { cmp::max(self.crit_pos, self.memory) };
            for i in start..needle.len() {
                if needle[i] != haystack[self.position + i] {
                    self.position += i - self.crit_pos + 1;
                    if !long_period {
                        self.memory = 0;
                    }
                    continue 'search;
                }
            }

            // See if the left part of the needle matches
            let start = if long_period { 0 } else { self.memory };
            for i in (start..self.crit_pos).rev() {
                if needle[i] != haystack[self.position + i] {
                    self.position += self.period;
                    if !long_period {
                        self.memory = needle.len() - self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            let match_pos = self.position;

            // Note: add self.period instead of needle.len() to have overlapping matches
            self.position += needle.len();
            if !long_period {
                self.memory = 0; // set to needle.len() - self.period for overlapping matches
            }

            return S::matching(match_pos, match_pos + needle.len());
        }
    }

    // Follows the ideas in `next()`.
    //
    // The definitions are symmetrical, with period(x) = period(reverse(x))
    // and local_period(u, v) = local_period(reverse(v), reverse(u)), so if (u, v)
    // is a critical factorization, so is (reverse(v), reverse(u)).
    //
    // For the reverse case we have computed a critical factorization x = u' v'
    // (field `crit_pos_back`). We need |u| < period(x) for the forward case and
    // thus |v'| < period(x) for the reverse.
    //
    // To search in reverse through the haystack, we search forward through
    // a reversed haystack with a reversed needle, matching first u' and then v'.
    #[inline]
    fn next_back<S>(&mut self, haystack: &[u8], needle: &[u8], long_period: bool) -> S::Output
    where
        S: TwoWayStrategy,
    {
        // `next_back()` uses `self.end` as its cursor -- so that `next()` and `next_back()`
        // are independent.
        let old_end = self.end;
        'search: loop {
            // Check that we have room to search in
            // end - needle.len() will wrap around when there is no more room,
            // but due to slice length limits it can never wrap all the way back
            // into the length of haystack.
            let front_byte = match haystack.get(self.end.wrapping_sub(needle.len())) {
                Some(&b) => b,
                None => {
                    self.end = 0;
                    return S::rejecting(0, old_end);
                }
            };

            if S::use_early_reject() && old_end != self.end {
                return S::rejecting(self.end, old_end);
            }

            // Quickly skip by large portions unrelated to our substring
            if !self.byteset_contains(front_byte) {
                self.end -= needle.len();
                if !long_period {
                    self.memory_back = needle.len();
                }
                continue 'search;
            }

            // See if the left part of the needle matches
            let crit = if long_period {
                self.crit_pos_back
            } else {
                cmp::min(self.crit_pos_back, self.memory_back)
            };
            for i in (0..crit).rev() {
                if needle[i] != haystack[self.end - needle.len() + i] {
                    self.end -= self.crit_pos_back - i;
                    if !long_period {
                        self.memory_back = needle.len();
                    }
                    continue 'search;
                }
            }

            // See if the right part of the needle matches
            let needle_end = if long_period { needle.len() } else { self.memory_back };
            for i in self.crit_pos_back..needle_end {
                if needle[i] != haystack[self.end - needle.len() + i] {
                    self.end -= self.period;
                    if !long_period {
                        self.memory_back = self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            let match_pos = self.end - needle.len();
            // Note: sub self.period instead of needle.len() to have overlapping matches
            self.end -= needle.len();
            if !long_period {
                self.memory_back = needle.len();
            }

            return S::matching(match_pos, match_pos + needle.len());
        }
    }

    // Compute the maximal suffix of `arr`.
    //
    // The maximal suffix is a possible critical factorization (u, v) of `arr`.
    //
    // Returns (`i`, `p`) where `i` is the starting index of v and `p` is the
    // period of v.
    //
    // `order_greater` determines if lexical order is `<` or `>`. Both
    // orders must be computed -- the ordering with the largest `i` gives
    // a critical factorization.
    //
    // For long period cases, the resulting period is not exact (it is too short).
    #[inline]
    fn maximal_suffix(arr: &[u8], order_greater: bool) -> (usize, usize) {
        let mut left = 0; // Corresponds to i in the paper
        let mut right = 1; // Corresponds to j in the paper
        let mut offset = 0; // Corresponds to k in the paper, but starting at 0
        // to match 0-based indexing.
        let mut period = 1; // Corresponds to p in the paper

        while let Some(&a) = arr.get(right + offset) {
            // `left` will be inbounds when `right` is.
            let b = arr[left + offset];
            if (a < b && !order_greater) || (a > b && order_greater) {
                // Suffix is smaller, period is entire prefix so far.
                right += offset + 1;
                offset = 0;
                period = right - left;
            } else if a == b {
                // Advance through repetition of the current period.
                if offset + 1 == period {
                    right += offset + 1;
                    offset = 0;
                } else {
                    offset += 1;
                }
            } else {
                // Suffix is larger, start over from current location.
                left = right;
                right += 1;
                offset = 0;
                period = 1;
            }
        }
        (left, period)
    }

    // Compute the maximal suffix of the reverse of `arr`.
    //
    // The maximal suffix is a possible critical factorization (u', v') of `arr`.
    //
    // Returns `i` where `i` is the starting index of v', from the back;
    // returns immediately when a period of `known_period` is reached.
    //
    // `order_greater` determines if lexical order is `<` or `>`. Both
    // orders must be computed -- the ordering with the largest `i` gives
    // a critical factorization.
    //
    // For long period cases, the resulting period is not exact (it is too short).
    fn reverse_maximal_suffix(arr: &[u8], known_period: usize, order_greater: bool) -> usize {
        let mut left = 0; // Corresponds to i in the paper
        let mut right = 1; // Corresponds to j in the paper
        let mut offset = 0; // Corresponds to k in the paper, but starting at 0
        // to match 0-based indexing.
        let mut period = 1; // Corresponds to p in the paper
        let n = arr.len();

        while right + offset < n {
            let a = arr[n - (1 + right + offset)];
            let b = arr[n - (1 + left + offset)];
            if (a < b && !order_greater) || (a > b && order_greater) {
                // Suffix is smaller, period is entire prefix so far.
                right += offset + 1;
                offset = 0;
                period = right - left;
            } else if a == b {
                // Advance through repetition of the current period.
                if offset + 1 == period {
                    right += offset + 1;
                    offset = 0;
                } else {
                    offset += 1;
                }
            } else {
                // Suffix is larger, start over from current location.
                left = right;
                right += 1;
                offset = 0;
                period = 1;
            }
            if period == known_period {
                break;
            }
        }
        debug_assert!(period <= known_period);
        left
    }
}

// TwoWayStrategy allows the algorithm to either skip non-matches as quickly
// as possible, or to work in a mode where it emits Rejects relatively quickly.
trait TwoWayStrategy {
    type Output;
    fn use_early_reject() -> bool;
    fn rejecting(a: usize, b: usize) -> Self::Output;
    fn matching(a: usize, b: usize) -> Self::Output;
}

/// Skip to match intervals as quickly as possible
enum MatchOnly {}

impl TwoWayStrategy for MatchOnly {
    type Output = Option<(usize, usize)>;

    #[inline]
    fn use_early_reject() -> bool {
        false
    }
    #[inline]
    fn rejecting(_a: usize, _b: usize) -> Self::Output {
        None
    }
    #[inline]
    fn matching(a: usize, b: usize) -> Self::Output {
        Some((a, b))
    }
}

/// Emit Rejects regularly
enum RejectAndMatch {}

impl TwoWayStrategy for RejectAndMatch {
    type Output = SearchStep;

    #[inline]
    fn use_early_reject() -> bool {
        true
    }
    #[inline]
    fn rejecting(a: usize, b: usize) -> Self::Output {
        SearchStep::Reject(a, b)
    }
    #[inline]
    fn matching(a: usize, b: usize) -> Self::Output {
        SearchStep::Match(a, b)
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
fn simd_contains(needle: &[u8], haystack: &[u8]) -> Option<bool> {
    debug_assert!(needle.len() > 1);

    use crate::ops::BitAnd;
    use crate::simd::cmp::SimdPartialEq;
    use crate::simd::{mask8x16 as Mask, u8x16 as Block};

    let first_probe = needle[0];
    let last_byte_offset = needle.len() - 1;

    // the offset used for the 2nd vector
    let second_probe_offset = if needle.len() == 2 {
        // never bail out on len=2 needles because the probes will fully cover them and have
        // no degenerate cases.
        1
    } else {
        // try a few bytes in case first and last byte of the needle are the same
        let Some(second_probe_offset) =
            (needle.len().saturating_sub(4)..needle.len()).rfind(|&idx| needle[idx] != first_probe)
        else {
            // fall back to other search methods if we can't find any different bytes
            // since we could otherwise hit some degenerate cases
            return None;
        };
        second_probe_offset
    };

    // do a naive search if the haystack is too small to fit
    if haystack.len() < Block::LEN + last_byte_offset {
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
        false
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
        let mask = both.to_bitmask() as u16;

        mask
    };

    let mut i = 0;
    let mut result = false;
    // The loop condition must ensure that there's enough headroom to read LANE bytes,
    // and not only at the current index but also at the index shifted by block_offset
    const UNROLL: usize = 4;
    while i + last_byte_offset + UNROLL * Block::LEN < haystack.len() && !result {
        let mut masks = [0u16; UNROLL];
        for j in 0..UNROLL {
            masks[j] = test_chunk(i + j * Block::LEN);
        }
        for j in 0..UNROLL {
            let mask = masks[j];
            if mask != 0 {
                result |= check_mask(i + j * Block::LEN, mask, result);
            }
        }
        i += UNROLL * Block::LEN;
    }
    while i + last_byte_offset + Block::LEN < haystack.len() && !result {
        let mask = test_chunk(i);
        if mask != 0 {
            result |= check_mask(i, mask, result);
        }
        i += Block::LEN;
    }

    // Process the tail that didn't fit into LANES-sized steps.
    // This simply repeats the same procedure but as right-aligned chunk instead
    // of a left-aligned one. The last byte must be exactly flush with the string end so
    // we don't miss a single byte or read out of bounds.
    let i = haystack.len() - last_byte_offset - Block::LEN;
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
