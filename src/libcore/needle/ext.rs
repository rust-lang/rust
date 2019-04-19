//! Extension functions which can be applied on any pairs of [`Haystack`]/[`Needle`].

use super::haystack::{Hay, Haystack, Span};
use super::needle::{
    Needle, Searcher, ReverseSearcher, DoubleEndedSearcher,
    Consumer, ReverseConsumer, DoubleEndedConsumer,
};
use crate::iter::FusedIterator;
use crate::ops::Range;
use crate::fmt;

macro_rules! generate_clone_and_debug {
    ($name:ident, $field:tt) => {
        impl<H, S> Clone for $name<H, S>
        where
            H: Haystack + Clone,
            S: Clone,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {
            fn clone(&self) -> Self {
                $name { $field: self.$field.clone() }
            }
            fn clone_from(&mut self, src: &Self) {
                self.$field.clone_from(&src.$field);
            }
        }

        impl<H, S> fmt::Debug for $name<H, S>
        where
            H: Haystack + fmt::Debug,
            S: fmt::Debug,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($name))
                    .field(&self.$field)
                    .finish()
            }
        }
    }
}

macro_rules! generate_pattern_iterators {
    {
        // Forward iterator
        forward:
            $(#[$forward_iterator_attribute:meta])*
            struct $forward_iterator:ident;

        // Reverse iterator
        reverse:
            $(#[$reverse_iterator_attribute:meta])*
            struct $reverse_iterator:ident;

        // Stability of all generated items
        stability:
            $(#[$common_stability_attribute:meta])*

        // Internal almost-iterator that is being delegated to
        internal:
            $internal_iterator:ident yielding ($iterty:ty);

        // Kind of delegation - either single ended or double ended
        delegate $($t:tt)*
    } => {
        $(#[$forward_iterator_attribute])*
        $(#[$common_stability_attribute])*
        pub struct $forward_iterator<H, S>($internal_iterator<H, S>)
        where
            H::Target: Hay, // FIXME: RFC 2089 or 2289
            H: Haystack;

        generate_clone_and_debug!($forward_iterator, 0);

        $(#[$common_stability_attribute])*
        impl<H, S> Iterator for $forward_iterator<H, S>
        where
            H: Haystack,
            S: Searcher<H::Target>,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next()
            }
        }

        $(#[$reverse_iterator_attribute])*
        $(#[$common_stability_attribute])*
        pub struct $reverse_iterator<H, S>($internal_iterator<H, S>)
        where
            H::Target: Hay, // FIXME: RFC 2089 or 2289
            H: Haystack;

        generate_clone_and_debug!($reverse_iterator, 0);

        $(#[$common_stability_attribute])*
        impl<H, S> Iterator for $reverse_iterator<H, S>
        where
            H: Haystack,
            S: ReverseSearcher<H::Target>,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                self.0.next_back()
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<H, S> FusedIterator for $forward_iterator<H, S>
        where
            H: Haystack,
            S: Searcher<H::Target>,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {}

        #[stable(feature = "fused", since = "1.26.0")]
        impl<H, S> FusedIterator for $reverse_iterator<H, S>
        where
            H: Haystack,
            S: ReverseSearcher<H::Target>,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {}

        generate_pattern_iterators!($($t)* with $(#[$common_stability_attribute])*,
                                                $forward_iterator,
                                                $reverse_iterator);
    };
    {
        double ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident
    } => {
        $(#[$common_stability_attribute])*
        impl<H, S> DoubleEndedIterator for $forward_iterator<H, S>
        where
            H: Haystack,
            S: DoubleEndedSearcher<H::Target>,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back()
            }
        }

        $(#[$common_stability_attribute])*
        impl<H, S> DoubleEndedIterator for $reverse_iterator<H, S>
        where
            H: Haystack,
            S: DoubleEndedSearcher<H::Target>,
            H::Target: Hay, // FIXME: RFC 2089 or 2289
        {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next()
            }
        }
    };
    {
        single ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident
    } => {}
}

//------------------------------------------------------------------------------
// Starts with / Ends with
//------------------------------------------------------------------------------

/// Returns `true` if the given needle matches a prefix of the haystack.
///
/// Returns `false` if it does not.
pub fn starts_with<H, P>(haystack: H, needle: P) -> bool
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_consumer().consume((*haystack).into()).is_some()
}

/// Returns `true` if the given needle matches a suffix of this haystack.
///
/// Returns `false` if it does not.
#[inline]
pub fn ends_with<H, P>(haystack: H, needle: P) -> bool
where
    H: Haystack,
    P: Needle<H>,
    P::Consumer: ReverseConsumer<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_consumer().rconsume((*haystack).into()).is_some()
}

//------------------------------------------------------------------------------
// Trim
//------------------------------------------------------------------------------

/// Returns a haystack slice with all prefixes that match the needle repeatedly removed.
#[inline]
pub fn trim_start<H, P>(haystack: H, needle: P) -> H
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    let range = {
        let hay = &*haystack;
        let start = needle.into_consumer().trim_start(hay);
        let end = hay.end_index();
        start..end
    };
    unsafe { haystack.slice_unchecked(range) }
}

/// Returns a haystack slice with all suffixes that match the needle repeatedly removed.
pub fn trim_end<H, P>(haystack: H, needle: P) -> H
where
    H: Haystack,
    P: Needle<H>,
    P::Consumer: ReverseConsumer<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    let range = {
        let hay = &*haystack;
        let start = hay.start_index();
        let end = needle.into_consumer().trim_end(hay);
        start..end
    };
    unsafe { haystack.slice_unchecked(range) }
}

/// Returns a haystack slice with all prefixes and suffixes that match the needle
/// repeatedly removed.
pub fn trim<H, P>(haystack: H, needle: P) -> H
where
    H: Haystack,
    P: Needle<H>,
    P::Consumer: DoubleEndedConsumer<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    let mut checker = needle.into_consumer();
    let range = {
        let hay = &*haystack;
        let end = checker.trim_end(hay);
        let hay = unsafe { Hay::slice_unchecked(hay, hay.start_index()..end) };
        let start = checker.trim_start(hay);
        start..end
    };
    unsafe { haystack.slice_unchecked(range) }
}

//------------------------------------------------------------------------------
// Matches
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct MatchesInternal<H, S>
where
    H: Haystack,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    searcher: S,
    rest: Span<H>,
}

impl<H, S> MatchesInternal<H, S>
where
    H: Haystack,
    S: Searcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next_spanned(&mut self) -> Option<Span<H>> {
        let rest = self.rest.take();
        let range = self.searcher.search(rest.borrow())?;
        let [_, middle, right] = unsafe { rest.split_around(range) };
        self.rest = right;
        Some(middle)
    }

    #[inline]
    fn next(&mut self) -> Option<H> {
        Some(Span::into(self.next_spanned()?))
    }
}

impl<H, S> MatchesInternal<H, S>
where
    H: Haystack,
    S: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next_back_spanned(&mut self) -> Option<Span<H>> {
        let rest = self.rest.take();
        let range = self.searcher.rsearch(rest.borrow())?;
        let [left, middle, _] = unsafe { rest.split_around(range) };
        self.rest = left;
        Some(middle)
    }

    #[inline]
    fn next_back(&mut self) -> Option<H> {
        Some(Span::into(self.next_back_spanned()?))
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the function [`matches`](fn.matches.html).
        struct Matches;
    reverse:
        /// Created with the function [`rmatches`](fn.rmatches.html).
        struct RMatches;
    stability:
    internal:
        MatchesInternal yielding (H);
    delegate double ended;
}

/// An iterator over the disjoint matches of the needle within the given haystack.
pub fn matches<H, P>(haystack: H, needle: P) -> Matches<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    Matches(MatchesInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
    })
}

/// An iterator over the disjoint matches of the needle within the haystack,
/// yielded in reverse order.
pub fn rmatches<H, P>(haystack: H, needle: P) -> RMatches<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    RMatches(MatchesInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
    })
}

/// Returns `true` if the given needle matches a sub-slice of the haystack.
///
/// Returns `false` if it does not.
pub fn contains<H, P>(haystack: H, needle: P) -> bool
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_searcher()
        .search((*haystack).into())
        .is_some()
}

//------------------------------------------------------------------------------
// MatchIndices
//------------------------------------------------------------------------------

struct MatchIndicesInternal<H, S>
where
    H: Haystack,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    inner: MatchesInternal<H, S>,
}

generate_clone_and_debug!(MatchIndicesInternal, inner);

impl<H, S> MatchIndicesInternal<H, S>
where
    H: Haystack,
    S: Searcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next(&mut self) -> Option<(<H::Target as Hay>::Index, H)> {
        let span = self.inner.next_spanned()?;
        let index = span.original_range().start;
        Some((index, Span::into(span)))
    }
}

impl<H, S> MatchIndicesInternal<H, S>
where
    H: Haystack,
    S: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next_back(&mut self) -> Option<(<H::Target as Hay>::Index, H)> {
        let span = self.inner.next_back_spanned()?;
        let index = span.original_range().start;
        Some((index, Span::into(span)))
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the function [`match_indices`](fn.match_indices.html).
        struct MatchIndices;
    reverse:
        /// Created with the function [`rmatch_indices`](fn.rmatch_indices.html).
        struct RMatchIndices;
    stability:
    internal:
        MatchIndicesInternal yielding ((<H::Target as Hay>::Index, H));
    delegate double ended;
}

/// An iterator over the disjoint matches of a needle within the haystack
/// as well as the index that the match starts at.
///
/// For matches of `needle` within `haystack` that overlap,
/// only the indices corresponding to the first match are returned.
pub fn match_indices<H, P>(haystack: H, needle: P) -> MatchIndices<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    MatchIndices(MatchIndicesInternal {
        inner: matches(haystack, needle).0,
    })
}

/// An iterator over the disjoint matches of a needle within the haystack,
/// yielded in reverse order along with the index of the match.
///
/// For matches of `needle` within `haystack` that overlap,
/// only the indices corresponding to the last match are returned.
pub fn rmatch_indices<H, P>(haystack: H, needle: P) -> RMatchIndices<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    RMatchIndices(MatchIndicesInternal {
        inner: rmatches(haystack, needle).0,
    })
}

/// Returns the start index of first slice of the haystack that matches the needle.
///
/// Returns [`None`] if the pattern doesn't match.
#[inline]
pub fn find<H, P>(haystack: H, needle: P) -> Option<<H::Target as Hay>::Index>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_searcher()
        .search((*haystack).into())
        .map(|r| r.start)
}

/// Returns the start index of last slice of the haystack that matches the needle.
///
/// Returns [`None`] if the pattern doesn't match.
pub fn rfind<H, P>(haystack: H, needle: P) -> Option<<H::Target as Hay>::Index>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_searcher()
        .rsearch((*haystack).into())
        .map(|r| r.start)
}

//------------------------------------------------------------------------------
// MatchRanges
//------------------------------------------------------------------------------

struct MatchRangesInternal<H, S>
where
    H: Haystack,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    inner: MatchesInternal<H, S>,
}

generate_clone_and_debug!(MatchRangesInternal, inner);

impl<H, S> MatchRangesInternal<H, S>
where
    H: Haystack,
    S: Searcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next(&mut self) -> Option<(Range<<H::Target as Hay>::Index>, H)> {
        let span = self.inner.next_spanned()?;
        let range = span.original_range();
        Some((range, Span::into(span)))
    }
}

impl<H, S> MatchRangesInternal<H, S>
where
    H: Haystack,
    S: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next_back(&mut self) -> Option<(Range<<H::Target as Hay>::Index>, H)> {
        let span = self.inner.next_back_spanned()?;
        let range = span.original_range();
        Some((range, Span::into(span)))
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the function [`match_ranges`](fn.match_ranges.html).
        struct MatchRanges;
    reverse:
        /// Created with the function [`rmatch_ranges`](fn.rmatch_ranges.html).
        struct RMatchRanges;
    stability:
    internal:
        MatchRangesInternal yielding ((Range<<H::Target as Hay>::Index>, H));
    delegate double ended;
}

/// An iterator over the disjoint matches of a needle within the haystack
/// as well as the index ranges of each match.
///
/// For matches of `needle` within `haystack` that overlap,
/// only the ranges corresponding to the first match are returned.
pub fn match_ranges<H, P>(haystack: H, needle: P) -> MatchRanges<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    MatchRanges(MatchRangesInternal {
        inner: matches(haystack, needle).0,
    })
}

/// An iterator over the disjoint matches of a needle within the haystack,
/// yielded in reverse order along with the index range of the match.
///
/// For matches of `needle` within `haystack` that overlap,
/// only the ranges corresponding to the last match are returned.
pub fn rmatch_ranges<H, P>(haystack: H, needle: P) -> RMatchRanges<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    RMatchRanges(MatchRangesInternal {
        inner: rmatches(haystack, needle).0,
    })
}

/// Returns the index range of first slice of the haystack that matches the needle.
///
/// Returns [`None`] if the pattern doesn't match.
pub fn find_range<H, P>(haystack: H, needle: P) -> Option<Range<<H::Target as Hay>::Index>>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_searcher()
        .search((*haystack).into())
}

/// Returns the start index of last slice of the haystack that matches the needle.
///
/// Returns [`None`] if the pattern doesn't match.
pub fn rfind_range<H, P>(haystack: H, needle: P) -> Option<Range<<H::Target as Hay>::Index>>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    needle.into_searcher()
        .rsearch((*haystack).into())
}

//------------------------------------------------------------------------------
// Split
//------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SplitInternal<H, S>
where
    H: Haystack,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    searcher: S,
    rest: Span<H>,
    finished: bool,
    allow_trailing_empty: bool,
}

impl<H, S> SplitInternal<H, S>
where
    H: Haystack,
    S: Searcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next(&mut self) -> Option<H> {
        if self.finished {
            return None;
        }

        let mut rest = self.rest.take();
        match self.searcher.search(rest.borrow()) {
            Some(subrange) => {
                let [left, _, right] = unsafe { rest.split_around(subrange) };
                self.rest = right;
                rest = left;
            }
            None => {
                self.finished = true;
                if !self.allow_trailing_empty && rest.is_empty() {
                    return None;
                }
            }
        }
        Some(Span::into(rest))
    }
}

impl<H, S> SplitInternal<H, S>
where
    H: Haystack,
    S: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next_back(&mut self) -> Option<H> {
        if self.finished {
            return None;
        }

        let rest = self.rest.take();
        let after = match self.searcher.rsearch(rest.borrow()) {
            Some(range) => {
                let [left, _, right] = unsafe { rest.split_around(range) };
                self.rest = left;
                right
            }
            None => {
                self.finished = true;
                rest
            }
        };

        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            if after.is_empty() {
                return self.next_back();
            }
        }

        Some(Span::into(after))
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the function [`split`](fn.split.html).
        struct Split;
    reverse:
        /// Created with the function [`rsplit`](fn.rsplit.html).
        struct RSplit;
    stability:
    internal:
        SplitInternal yielding (H);
    delegate double ended;
}

generate_pattern_iterators! {
    forward:
        /// Created with the function [`split_terminator`](fn.split_terminator.html).
        struct SplitTerminator;
    reverse:
        /// Created with the function [`rsplit_terminator`](fn.rsplit_terminator.html).
        struct RSplitTerminator;
    stability:
    internal:
        SplitInternal yielding (H);
    delegate double ended;
}

/// An iterator over slices of the haystack, separated by parts matched by the needle.
pub fn split<H, P>(haystack: H, needle: P) -> Split<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    Split(SplitInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
        finished: false,
        allow_trailing_empty: true,
    })
}

/// An iterator over slices of the haystack, separated by parts matched by the needle
/// and yielded in reverse order.
pub fn rsplit<H, P>(haystack: H, needle: P) -> RSplit<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    RSplit(SplitInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
        finished: false,
        allow_trailing_empty: true,
    })
}

/// An iterator over slices of the haystack, separated by parts matched by the needle.
///
/// Equivalent to [`split`](fn.split.html), except that the trailing slice is skipped if empty.
///
/// This method can be used for haystack data that is *terminated*,
/// rather than *separated* by a needle.
pub fn split_terminator<H, P>(haystack: H, needle: P) -> SplitTerminator<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    SplitTerminator(SplitInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
        finished: false,
        allow_trailing_empty: false,
    })
}

/// An iterator over slices of the haystack, separated by parts matched by the needle
/// and yielded in reverse order.
///
/// Equivalent to [`rsplit`](fn.rsplit.html), except that the trailing slice is skipped if empty.
///
/// This method can be used for haystack data that is *terminated*,
/// rather than *separated* by a needle.
pub fn rsplit_terminator<H, P>(haystack: H, needle: P) -> RSplitTerminator<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    RSplitTerminator(SplitInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
        finished: false,
        allow_trailing_empty: false,
    })
}

//------------------------------------------------------------------------------
// SplitN
//------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SplitNInternal<H, S>
where
    H: Haystack,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    searcher: S,
    rest: Span<H>,
    n: usize,
}

impl<H, S> SplitNInternal<H, S>
where
    H: Haystack,
    S: Searcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next(&mut self) -> Option<H> {
        let mut rest = self.rest.take();
        match self.n {
            0 => {
                return None;
            }
            1 => {
                self.n = 0;
            }
            n => {
                match self.searcher.search(rest.borrow()) {
                    Some(range) => {
                        let [left, _, right] = unsafe { rest.split_around(range) };
                        self.n = n - 1;
                        self.rest = right;
                        rest = left;
                    }
                    None => {
                        self.n = 0;
                    }
                }
            }
        }
        Some(Span::into(rest))
    }
}

impl<H, S> SplitNInternal<H, S>
where
    H: Haystack,
    S: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    #[inline]
    fn next_back(&mut self) -> Option<H> {
        let mut rest = self.rest.take();
        match self.n {
            0 => {
                return None;
            }
            1 => {
                self.n = 0;
            }
            n => {
                match self.searcher.rsearch(rest.borrow()) {
                    Some(range) => {
                        let [left, _, right] = unsafe { rest.split_around(range) };
                        self.n = n - 1;
                        self.rest = left;
                        rest = right;
                    }
                    None => {
                        self.n = 0;
                    }
                }
            }
        }
        Some(Span::into(rest))
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the function [`splitn`](fn.splitn.html).
        struct SplitN;
    reverse:
        /// Created with the function [`rsplitn`](fn.rsplitn.html).
        struct RSplitN;
    stability:
    internal:
        SplitNInternal yielding (H);
    delegate single ended;
}

/// An iterator over slices of the given haystack, separated by a needle,
/// restricted to returning at most `n` items.
///
/// If `n` slices are returned,
/// the last slice (the `n`th slice) will contain the remainder of the haystack.
pub fn splitn<H, P>(haystack: H, n: usize, needle: P) -> SplitN<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    SplitN(SplitNInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
        n,
    })
}

/// An iterator over slices of the given haystack, separated by a needle,
/// starting from the end of the haystack, restricted to returning at most `n` items.
///
/// If `n` slices are returned,
/// the last slice (the `n`th slice) will contain the remainder of the haystack.
pub fn rsplitn<H, P>(haystack: H, n: usize, needle: P) -> RSplitN<H, P::Searcher>
where
    H: Haystack,
    P: Needle<H>,
    P::Searcher: ReverseSearcher<H::Target>,
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    RSplitN(SplitNInternal {
        searcher: needle.into_searcher(),
        rest: haystack.into(),
        n,
    })
}

//------------------------------------------------------------------------------
// Replace
//------------------------------------------------------------------------------

/// Replaces all matches of a needle with another haystack.
pub fn replace_with<H, P, F, W>(src: H, from: P, mut replacer: F, mut writer: W)
where
    H: Haystack,
    P: Needle<H>,
    F: FnMut(H) -> H,
    W: FnMut(H),
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    let mut searcher = from.into_searcher();
    let mut src = Span::from(src);
    while let Some(range) = searcher.search(src.borrow()) {
        let [left, middle, right] = unsafe { src.split_around(range) };
        writer(Span::into(left));
        writer(replacer(Span::into(middle)));
        src = right;
    }
    writer(Span::into(src));
}

/// Replaces first `n` matches of a needle with another haystack.
pub fn replacen_with<H, P, F, W>(src: H, from: P, mut replacer: F, mut n: usize, mut writer: W)
where
    H: Haystack,
    P: Needle<H>,
    F: FnMut(H) -> H,
    W: FnMut(H),
    H::Target: Hay, // FIXME: RFC 2089 or 2289
{
    let mut searcher = from.into_searcher();
    let mut src = Span::from(src);
    loop {
        if n == 0 {
            break;
        }
        n -= 1;
        if let Some(range) = searcher.search(src.borrow()) {
            let [left, middle, right] = unsafe { src.split_around(range) };
            writer(Span::into(left));
            writer(replacer(Span::into(middle)));
            src = right;
        } else {
            break;
        }
    }
    writer(Span::into(src));
}
