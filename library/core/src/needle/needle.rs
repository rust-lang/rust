use super::haystack::{Hay, Haystack, Span};

use crate::ops::Range;

/// A searcher, for searching a [`Needle`] from a [`Hay`].
///
/// This trait provides methods for searching for non-overlapping matches of a
/// needle starting from the front (left) of a hay.
///
/// # Safety
///
/// This trait is marked unsafe because the range returned by its methods are
/// required to lie on valid codeword boundaries in the haystack. This enables
/// users of this trait to slice the haystack without additional runtime checks.
///
/// # Examples
///
/// Implement a searcher and consumer which matches `b"Aaaa"` from a byte string.
///
/// ```rust
/// #![feature(needle)]
/// use std::needle::*;
/// use std::ops::Range;
///
/// // The searcher for searching `b"Aaaa"`, using naive search.
/// // We are going to use this as a needle too.
/// struct Aaaa;
///
/// unsafe impl Searcher<[u8]> for Aaaa {
///     // search for an `b"Aaaa"` in the middle of the string, returns its range.
///     fn search(&mut self, span: Span<&[u8]>) -> Option<Range<usize>> {
///         let (hay, range) = span.into_parts();
///
///         let start = range.start;
///         for (i, window) in hay[range].windows(4).enumerate() {
///             if *window == b"Aaaa"[..] {
///                 // remember to include the range offset
///                 return Some((start + i)..(start + i + 4));
///             }
///         }
///
///         None
///     }
/// }
///
/// unsafe impl Consumer<[u8]> for Aaaa {
///     // checks if an `b"Aaaa" is at the beginning of the string, returns the end index.
///     fn consume(&mut self, span: Span<&[u8]>) -> Option<usize> {
///         let (hay, range) = span.into_parts();
///         let end = range.start.checked_add(4)?;
///         if end <= range.end && hay[range.start..end] == b"Aaaa"[..] {
///             Some(end)
///         } else {
///             None
///         }
///     }
/// }
///
/// impl<H: Haystack<Target = [u8]>> Needle<H> for Aaaa {
///     type Searcher = Self;
///     type Consumer = Self;
///     fn into_searcher(self) -> Self { self }
///     fn into_consumer(self) -> Self { self }
/// }
///
/// // test with some standard algorithms.
/// let haystack = &b"Aaaaa!!!Aaa!!!Aaaaaaaaa!!!"[..];
/// assert_eq!(
///     ext::split(haystack, Aaaa).collect::<Vec<_>>(),
///     vec![
///         &b""[..],
///         &b"a!!!Aaa!!!"[..],
///         &b"aaaaa!!!"[..],
///     ]
/// );
/// assert_eq!(
///     ext::match_ranges(haystack, Aaaa).collect::<Vec<_>>(),
///     vec![
///         (0..4, &b"Aaaa"[..]),
///         (14..18, &b"Aaaa"[..]),
///     ]
/// );
/// assert_eq!(
///     ext::trim_start(haystack, Aaaa),
///     &b"a!!!Aaa!!!Aaaaaaaaa!!!"[..]
/// );
/// ```
pub unsafe trait Searcher<A: Hay + ?Sized> {
    /// Searches for the first range which the needle can be found in the span.
    ///
    /// This method is used to support the following standard algorithms:
    ///
    /// * [`matches`](super::ext::matches)
    /// * [`contains`](super::ext::contains)
    /// * [`match_indices`](super::ext::match_indices)
    /// * [`find`](super::ext::find)
    /// * [`match_ranges`](super::ext::match_ranges)
    /// * [`find_range`](super::ext::find_range)
    /// * [`split`](super::ext::split)
    /// * [`split_terminator`](super::ext::split_terminator)
    /// * [`splitn`](super::ext::splitn)
    /// * [`replace_with`](super::ext::replace_with)
    /// * [`replacen_with`](super::ext::replacen_with)
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](Span::into_parts). The range returned
    /// by this method
    /// should be relative to the hay and must be contained within the
    /// restricted range from the span.
    ///
    /// If the needle is not found, this method should return `None`.
    ///
    /// The reason this method takes a `Span<&A>` instead of just `&A` is
    /// because some needles need context information provided by
    /// the position of the current slice and the content around the slice.
    /// Regex components like the start-/end-of-text anchors `^`/`$`
    /// and word boundary `\b` are primary examples.
    fn search(&mut self, span: Span<&A>) -> Option<Range<A::Index>>;
}

/// A consumer, for searching a [`Needle`] from a [`Hay`] anchored at the
/// beginnning.
///
/// This trait provides methods for matching a needle anchored at the beginning
/// of a hay.
///
/// See documentation of [`Searcher`] for an example.
///
/// # Safety
///
/// This trait is marked unsafe because the range returned by its methods are
/// required to lie on valid codeword boundaries in the haystack. This enables
/// users of this trait to slice the haystack without additional runtime checks.
pub unsafe trait Consumer<A: Hay + ?Sized> {
    /// Checks if the needle can be found at the beginning of the span.
    ///
    /// This method is used to implement the standard algorithm
    /// [`starts_with()`](super::ext::starts_with) as well as providing the default
    /// implementation for [`.trim_start()`](Consumer::trim_start).
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](Span::into_parts). If a needle can be
    /// found starting at `range.start`, this method should return the end index
    /// of the needle relative to the hay.
    ///
    /// If the needle cannot be found at the beginning of the span, this method
    /// should return `None`.
    fn consume(&mut self, span: Span<&A>) -> Option<A::Index>;

    /// Repeatedly removes prefixes of the hay which matches the needle.
    ///
    /// This method is used to implement the standard algorithm
    /// [`trim_start()`](super::ext::trim_start).
    ///
    /// Returns the start index of the slice after all prefixes are removed.
    ///
    /// A fast generic implementation in terms of
    /// [`.consume()`](Consumer::consume) is provided by default. Nevertheless,
    /// many needles allow a higher-performance specialization.
    #[inline]
    fn trim_start(&mut self, hay: &A) -> A::Index {
        let mut offset = hay.start_index();
        let mut span = Span::from(hay);
        while let Some(pos) = self.consume(span.clone()) {
            offset = pos;
            let (hay, range) = span.into_parts();
            if pos == range.start {
                break;
            }
            // SAFETY: span's range is guaranteed to be valid for the haystack.
            span = unsafe { Span::from_parts(hay, pos..range.end) };
        }
        offset
    }
}

/// A searcher which can be searched from the end.
///
/// This trait provides methods for searching for non-overlapping matches of a
/// needle starting from the back (right) of a hay.
///
/// # Safety
///
/// This trait is marked unsafe because the range returned by its methods are
/// required to lie on valid codeword boundaries in the haystack. This enables
/// users of this trait to slice the haystack without additional runtime checks.
pub unsafe trait ReverseSearcher<A: Hay + ?Sized>: Searcher<A> {
    /// Searches for the last range which the needle can be found in the span.
    ///
    /// This method is used to support the following standard algorithms:
    ///
    /// * [`rmatches`](super::ext::rmatches)
    /// * [`rmatch_indices`](super::ext::rmatch_indices)
    /// * [`rfind`](super::ext::find)
    /// * [`rmatch_ranges`](super::ext::rmatch_ranges)
    /// * [`rfind_range`](super::ext::rfind_range)
    /// * [`rsplit`](super::ext::rsplit)
    /// * [`rsplit_terminator`](super::ext::rsplit_terminator)
    /// * [`rsplitn`](super::ext::rsplitn)
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](Span::into_parts). The returned range
    /// should be relative to the hay and must be contained within the
    /// restricted range from the span.
    ///
    /// If the needle is not found, this method should return `None`.
    fn rsearch(&mut self, span: Span<&A>) -> Option<Range<A::Index>>;
}

/// A consumer which can be searched from the end.
///
/// This trait provides methods for matching a needle anchored at the end of a
/// hay.
///
/// # Safety
///
/// This trait is marked unsafe because the range returned by its methods are
/// required to lie on valid codeword boundaries in the haystack. This enables
/// users of this trait to slice the haystack without additional runtime checks.
pub unsafe trait ReverseConsumer<A: Hay + ?Sized>: Consumer<A> {
    /// Checks if the needle can be found at the end of the span.
    ///
    /// This method is used to implement the standard algorithm
    /// [`ends_with()`](super::ext::ends_with) as well as providing the default
    /// implementation for [`.trim_end()`](ReverseConsumer::trim_end).
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](Span::into_parts). If a needle can be
    /// found ending at `range.end`, this method should return the start index
    /// of the needle relative to the hay.
    ///
    /// If the needle cannot be found at the end of the span, this method
    /// should return `None`.
    fn rconsume(&mut self, hay: Span<&A>) -> Option<A::Index>;

    /// Repeatedly removes suffixes of the hay which matches the needle.
    ///
    /// This method is used to implement the standard algorithm
    /// [`trim_end()`](super::ext::trim_end).
    ///
    /// A fast generic implementation in terms of
    /// [`.rconsume()`](ReverseConsumer::rconsume) is provided by default.
    /// Nevertheless, many needles allow a higher-performance specialization.
    #[inline]
    fn trim_end(&mut self, hay: &A) -> A::Index {
        let mut offset = hay.end_index();
        let mut span = Span::from(hay);
        while let Some(pos) = self.rconsume(span.clone()) {
            offset = pos;
            let (hay, range) = span.into_parts();
            if pos == range.end {
                break;
            }
            // SAFETY: span's range is guaranteed to be valid for the haystack.
            span = unsafe { Span::from_parts(hay, range.start..pos) };
        }
        offset
    }
}

/// A searcher which can be searched from both end with consistent results.
///
/// Implementing this marker trait enables the following standard algorithms to
/// return [`DoubleEndedIterator`](crate::iter::DoubleEndedIterator)s:
///
/// * [`matches`](super::ext::matches) /
///     [`rmatches`](super::ext::rmatches)
/// * [`match_indices`](super::ext::match_indices) /
///     [`rmatch_indices`](super::ext::rmatch_indices)
/// * [`match_ranges`](super::ext::match_ranges) /
///     [`rmatch_ranges`](super::ext::rmatch_ranges)
/// * [`split`](super::ext::split) /
///     [`rsplit`](super::ext::rsplit)
/// * [`split_terminator`](super::ext::split_terminator) /
///     [`rsplit_terminator`](super::ext::rsplit_terminator)
/// * [`splitn`](super::ext::splitn) /
///     [`rsplitn`](super::ext::rsplitn)
pub unsafe trait DoubleEndedSearcher<A: Hay + ?Sized>: ReverseSearcher<A> {}

/// A consumer which can be searched from both end with consistent results.
///
/// It is used to support the following standard algorithm:
///
/// * [`trim`](super::ext::trim)
///
/// The `trim` function is implemented by calling
/// [`trim_start`](super::ext::trim_start) and [`trim_end`](super::ext::trim_end)
/// together. This trait encodes the fact that we can call these two functions in any order.
pub unsafe trait DoubleEndedConsumer<A: Hay + ?Sized>: ReverseConsumer<A> {}

/// A needle, a type which can be converted into a searcher.
///
/// When using search algorithms like [`split()`](super::ext::split), users will
/// search with a `Needle` e.g. a `&str`. A needle is usually stateless,
/// however for efficient searching, we often need some preprocessing and
/// maintain a mutable state. The preprocessed structure is called the
/// [`Searcher`] of this needle.
///
/// The relationship between `Searcher` and `Needle` is similar to `Iterator`
/// and `IntoIterator`.
pub trait Needle<H: Haystack>: Sized
where
    H::Target: Hay,
{
    /// The searcher associated with this needle.
    type Searcher: Searcher<H::Target>;

    /// The consumer associated with this needle.
    type Consumer: Consumer<H::Target>;

    /// Produces a searcher for this needle.
    fn into_searcher(self) -> Self::Searcher;

    /// Produces a consumer for this needle.
    ///
    /// Usually a consumer and a searcher can be the same type.
    /// Some needles may require different types
    /// when the two need different optimization strategies. String searching
    /// is an example of this: we use the Two-Way Algorithm when searching for
    /// substrings, which needs to preprocess the needle. However this is
    /// irrelevant for consuming, which only needs to check for string equality
    /// once. Therefore the Consumer for a string would be a distinct type
    /// using naive search.
    fn into_consumer(self) -> Self::Consumer;
}

/// Searcher of an empty needle.
///
/// This searcher will find all empty subslices between any codewords in a
/// haystack.
#[derive(Clone, Debug, Default)]
pub struct EmptySearcher {
    consumed_start: bool,
    consumed_end: bool,
}

unsafe impl<A: Hay + ?Sized> Searcher<A> for EmptySearcher {
    #[inline]
    fn search(&mut self, span: Span<&A>) -> Option<Range<A::Index>> {
        let (hay, range) = span.into_parts();
        let start = if !self.consumed_start {
            self.consumed_start = true;
            range.start
        } else if range.start == range.end {
            return None;
        } else {
            // SAFETY: span's range is guaranteed to be valid for the haystack.
            unsafe { hay.next_index(range.start) }
        };
        Some(start..start)
    }
}

unsafe impl<A: Hay + ?Sized> Consumer<A> for EmptySearcher {
    #[inline]
    fn consume(&mut self, span: Span<&A>) -> Option<A::Index> {
        let (_, range) = span.into_parts();
        Some(range.start)
    }

    #[inline]
    fn trim_start(&mut self, hay: &A) -> A::Index {
        hay.start_index()
    }
}

unsafe impl<A: Hay + ?Sized> ReverseSearcher<A> for EmptySearcher {
    #[inline]
    fn rsearch(&mut self, span: Span<&A>) -> Option<Range<A::Index>> {
        let (hay, range) = span.into_parts();
        let end = if !self.consumed_end {
            self.consumed_end = true;
            range.end
        } else if range.start == range.end {
            return None;
        } else {
            // SAFETY: span's range is guaranteed to be valid for the haystack.
            unsafe { hay.prev_index(range.end) }
        };
        Some(end..end)
    }
}

unsafe impl<A: Hay + ?Sized> ReverseConsumer<A> for EmptySearcher {
    #[inline]
    fn rconsume(&mut self, span: Span<&A>) -> Option<A::Index> {
        let (_, range) = span.into_parts();
        Some(range.end)
    }

    #[inline]
    fn trim_end(&mut self, hay: &A) -> A::Index {
        hay.end_index()
    }
}

unsafe impl<A: Hay + ?Sized> DoubleEndedSearcher<A> for EmptySearcher {}
unsafe impl<A: Hay + ?Sized> DoubleEndedConsumer<A> for EmptySearcher {}
