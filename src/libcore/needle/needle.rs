use super::haystack::{Haystack, Hay, Span};

use crate::ops::Range;

/// A searcher, for searching a [`Needle`](trait.Needle.html) from a
/// [`Hay`](trait.Hay.html).
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
    /// * [`matches`](ext/fn.matches.html)
    /// * [`contains`](ext/fn.contains.html)
    /// * [`match_indices`](ext/fn.match_indices.html)
    /// * [`find`](ext/fn.find.html)
    /// * [`match_ranges`](ext/fn.match_ranges.html)
    /// * [`find_range`](ext/fn.find_range.html)
    /// * [`split`](ext/fn.split.html)
    /// * [`split_terminator`](ext/fn.split_terminator.html)
    /// * [`splitn`](ext/fn.splitn.html)
    /// * [`replace_with`](ext/fn.replace_with.html)
    /// * [`replacen_with`](ext/fn.replacen_with.html)
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](struct.Span.html#method.into_parts).
    /// The range returned by this method should be relative to the hay and
    /// must be contained within the restricted range from the span.
    ///
    /// If the needle is not found, this method should return `None`.
    ///
    /// The reason this method takes a `Span<&A>` instead of just `&A` is
    /// because some needles need context information provided by
    /// the position of the current slice and the content around the slice.
    /// Regex components like the start-/end-of-text anchors `^`/`$`
    /// and word boundary `\b` are primary examples.
    ///
    /// # Examples
    ///
    /// Search for the locations of a substring inside a string, using the
    /// searcher primitive.
    ///
    /// ```
    /// #![feature(needle)]
    /// use std::needle::{Searcher, Needle, Span};
    ///
    /// let mut searcher = Needle::<&str>::into_searcher("::");
    /// let span = Span::from("lion::tiger::leopard");
    /// //                     ^   ^      ^        ^
    /// // string indices:     0   4     11       20
    ///
    /// // found the first "::".
    /// assert_eq!(searcher.search(span.clone()), Some(4..6));
    ///
    /// // slice the span to skip the first match.
    /// let span = unsafe { span.slice_unchecked(6..20) };
    ///
    /// // found the second "::".
    /// assert_eq!(searcher.search(span.clone()), Some(11..13));
    ///
    /// // should find nothing now.
    /// let span = unsafe { span.slice_unchecked(13..20) };
    /// assert_eq!(searcher.search(span.clone()), None);
    /// ```
    fn search(&mut self, span: Span<&A>) -> Option<Range<A::Index>>;
}

/// A consumer, for searching a [`Needle`](trait.Needle.html) from a
/// [`Hay`](trait.Hay.html) anchored at the beginnning.
///
/// This trait provides methods for matching a needle anchored at the beginning
/// of a hay.
///
/// See documentation of [`Searcher`](trait.Searcher.html) for an example.
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
    /// [`starts_with()`](ext/fn.starts_with.html) as well as providing
    /// the default implementation for [`.trim_start()`](#method.trim_start).
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](struct.Span.html#method.into_parts).
    /// If a needle can be found starting at `range.start`, this method should
    /// return the end index of the needle relative to the hay.
    ///
    /// If the needle cannot be found at the beginning of the span, this method
    /// should return `None`.
    ///
    /// # Examples
    ///
    /// Consumes ASCII characters from the beginning.
    ///
    /// ```
    /// #![feature(needle)]
    /// use std::needle::{Consumer, Needle, Span};
    ///
    /// let mut consumer = Needle::<&str>::into_consumer(|c: char| c.is_ascii());
    /// let span = Span::from("Hi😋!!");
    ///
    /// // consumes the first ASCII character
    /// assert_eq!(consumer.consume(span.clone()), Some(1));
    ///
    /// // slice the span to skip the first match.
    /// let span = unsafe { span.slice_unchecked(1..8) };
    ///
    /// // matched the second ASCII character
    /// assert_eq!(consumer.consume(span.clone()), Some(2));
    ///
    /// // should match nothing now.
    /// let span = unsafe { span.slice_unchecked(2..8) };
    /// assert_eq!(consumer.consume(span.clone()), None);
    /// ```
    fn consume(&mut self, span: Span<&A>) -> Option<A::Index>;

    /// Repeatedly removes prefixes of the hay which matches the needle.
    ///
    /// This method is used to implement the standard algorithm
    /// [`trim_start()`](ext/fn.trim_start.html).
    ///
    /// Returns the start index of the slice after all prefixes are removed.
    ///
    /// A fast generic implementation in terms of
    /// [`.consume()`](#method.consume) is provided by default. Nevertheless,
    /// many needles allow a higher-performance specialization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(needle)]
    /// use std::needle::{Consumer, Needle};
    ///
    /// let mut consumer = Needle::<&str>::into_consumer('x');
    /// assert_eq!(consumer.trim_start("xxxyy"), 3);
    ///
    /// let mut consumer = Needle::<&str>::into_consumer('x');
    /// assert_eq!(consumer.trim_start("yyxxx"), 0);
    /// ```
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
    /// * [`rmatches`](ext/fn.rmatches.html)
    /// * [`rmatch_indices`](ext/fn.rmatch_indices.html)
    /// * [`rfind`](ext/fn.find.html)
    /// * [`rmatch_ranges`](ext/fn.rmatch_ranges.html)
    /// * [`rfind_range`](ext/fn.rfind_range.html)
    /// * [`rsplit`](ext/fn.rsplit.html)
    /// * [`rsplit_terminator`](ext/fn.rsplit_terminator.html)
    /// * [`rsplitn`](ext/fn.rsplitn.html)
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](struct.Span.html#method.into_parts).
    /// The returned range should be relative to the hay and must be contained
    /// within the restricted range from the span.
    ///
    /// If the needle is not found, this method should return `None`.
    ///
    /// # Examples
    ///
    /// Search for the locations of a substring inside a string, using the
    /// searcher primitive.
    ///
    /// ```
    /// #![feature(needle)]
    /// use std::needle::{ReverseSearcher, Needle, Span};
    ///
    /// let mut searcher = Needle::<&str>::into_searcher("::");
    /// let span = Span::from("lion::tiger::leopard");
    /// //                     ^   ^      ^
    /// // string indices:     0   4     11
    ///
    /// // found the last "::".
    /// assert_eq!(searcher.rsearch(span.clone()), Some(11..13));
    ///
    /// // slice the span to skip the last match.
    /// let span = unsafe { span.slice_unchecked(0..11) };
    ///
    /// // found the second to last "::".
    /// assert_eq!(searcher.rsearch(span.clone()), Some(4..6));
    ///
    /// // should found nothing now.
    /// let span = unsafe { span.slice_unchecked(0..4) };
    /// assert_eq!(searcher.rsearch(span.clone()), None);
    /// ```
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
    /// [`ends_with()`](ext/fn.ends_with.html) as well as providing the default
    /// implementation for [`.trim_end()`](#method.trim_end).
    ///
    /// The hay and the restricted range for searching can be recovered by
    /// calling `span`[`.into_parts()`](struct.Span.html#method.into_parts).
    /// If a needle can be found ending at `range.end`, this method should
    /// return the start index of the needle relative to the hay.
    ///
    /// If the needle cannot be found at the end of the span, this method
    /// should return `None`.
    ///
    /// # Examples
    ///
    /// Consumes ASCII characters from the end.
    ///
    /// ```
    /// #![feature(needle)]
    /// use std::needle::{ReverseConsumer, Needle, Span};
    ///
    /// let mut consumer = Needle::<&str>::into_consumer(|c: char| c.is_ascii());
    /// let span = Span::from("Hi😋!!");
    ///
    /// // consumes the last ASCII character
    /// assert_eq!(consumer.rconsume(span.clone()), Some(7));
    ///
    /// // slice the span to skip the first match.
    /// let span = unsafe { span.slice_unchecked(0..7) };
    ///
    /// // matched the second to last ASCII character
    /// assert_eq!(consumer.rconsume(span.clone()), Some(6));
    ///
    /// // should match nothing now.
    /// let span = unsafe { span.slice_unchecked(0..6) };
    /// assert_eq!(consumer.rconsume(span.clone()), None);
    /// ```
    fn rconsume(&mut self, hay: Span<&A>) -> Option<A::Index>;

    /// Repeatedly removes suffixes of the hay which matches the needle.
    ///
    /// This method is used to implement the standard algorithm
    /// [`trim_end()`](ext/fn.trim_end.html).
    ///
    /// A fast generic implementation in terms of
    /// [`.rconsume()`](#method.rconsume) is provided by default.
    /// Nevertheless, many needles allow a higher-performance specialization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(needle)]
    /// use std::needle::{ReverseConsumer, Needle};
    ///
    /// let mut consumer = Needle::<&str>::into_consumer('x');
    /// assert_eq!(consumer.trim_end("yyxxx"), 2);
    ///
    /// let mut consumer = Needle::<&str>::into_consumer('x');
    /// assert_eq!(consumer.trim_end("xxxyy"), 5);
    /// ```
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
            span = unsafe { Span::from_parts(hay, range.start..pos) };
        }
        offset
    }
}

/// A searcher which can be searched from both end with consistent results.
///
/// Implementing this marker trait enables the following standard algorithms to
/// return [`DoubleEndedIterator`](../iter/trait.DoubleEndedIterator.html)s:
///
/// * [`matches`](ext/fn.matches.html) /
///     [`rmatches`](ext/fn.rmatches.html)
/// * [`match_indices`](ext/fn.match_indices.html) /
///     [`rmatch_indices`](ext/fn.rmatch_indices.html)`
/// * [`match_ranges`](ext/fn.match_ranges.html) /
///     [`rmatch_ranges`](ext/fn.rmatch_ranges.html)
/// * [`split`](ext/fn.split.html) /
///     [`rsplit`](ext/fn.rsplit.html)
/// * [`split_terminator`](ext/fn.split_terminator.html) /
///     [`rsplit_terminator`](ext/fn.rsplit_terminator.html)
/// * [`splitn`](ext/fn.splitn.html) /
///     [`rsplitn`](ext/fn.rsplitn.html)
///
/// # Examples
///
/// The searcher of a character implements `DoubleEndedSearcher`, while that of
/// a string does not.
///
/// `match_indices` and `rmatch_indices` are reverse of each other only for a
/// `DoubleEndedSearcher`.
///
/// ```rust
/// #![feature(needle)]
/// use std::needle::ext::{match_indices, rmatch_indices};
///
/// // `match_indices` and `rmatch_indices` are exact reverse of each other for a `char` needle.
/// let forward = match_indices("xxxxx", 'x').collect::<Vec<_>>();
/// let mut rev_backward = rmatch_indices("xxxxx", 'x').collect::<Vec<_>>();
/// rev_backward.reverse();
///
/// assert_eq!(forward, vec![(0, "x"), (1, "x"), (2, "x"), (3, "x"), (4, "x")]);
/// assert_eq!(rev_backward, vec![(0, "x"), (1, "x"), (2, "x"), (3, "x"), (4, "x")]);
/// assert_eq!(forward, rev_backward);
///
/// // this property does not exist on a `&str` needle in general.
/// let forward = match_indices("xxxxx", "xx").collect::<Vec<_>>();
/// let mut rev_backward = rmatch_indices("xxxxx", "xx").collect::<Vec<_>>();
/// rev_backward.reverse();
///
/// assert_eq!(forward, vec![(0, "xx"), (2, "xx")]);
/// assert_eq!(rev_backward, vec![(1, "xx"), (3, "xx")]);
/// assert_ne!(forward, rev_backward);
/// ```
pub unsafe trait DoubleEndedSearcher<A: Hay + ?Sized>: ReverseSearcher<A> {}

/// A consumer which can be searched from both end with consistent results.
///
/// It is used to support the following standard algorithm:
///
/// * [`trim`](ext/fn.trim.html)
///
/// The `trim` function is implemented by calling
/// [`trim_start`](ext/fn.trim_start.html) and [`trim_end`](ext/fn.trim_end.html)
/// together. This trait encodes the fact that we can call these two functions in any order.
///
/// # Examples
///
/// The consumer of a character implements `DoubleEndedConsumer`, while that of
/// a string does not. `trim` is implemented only for a `DoubleEndedConsumer`.
///
/// ```rust
/// #![feature(needle)]
/// use std::needle::ext::{trim_start, trim_end, trim};
///
/// // for a `char`, we get the same trim result no matter which function is called first.
/// let trim_start_first = trim_end(trim_start("xyxyx", 'x'), 'x');
/// let trim_end_first = trim_start(trim_end("xyxyx", 'x'), 'x');
/// let trim_together = trim("xyxyx", 'x');
/// assert_eq!(trim_start_first, "yxy");
/// assert_eq!(trim_end_first, "yxy");
/// assert_eq!(trim_together, "yxy");
///
/// // this property does not exist for a `&str` in general.
/// let trim_start_first = trim_end(trim_start("xyxyx", "xyx"), "xyx");
/// let trim_end_first = trim_start(trim_end("xyxyx", "xyx"), "xyx");
/// // let trim_together = trim("xyxyx", 'x'); // cannot be defined
/// assert_eq!(trim_start_first, "yx");
/// assert_eq!(trim_end_first, "xy");
/// // assert_eq!(trim_together, /*????*/); // cannot be defined
/// ```
pub unsafe trait DoubleEndedConsumer<A: Hay + ?Sized>: ReverseConsumer<A> {}

/// A needle, a type which can be converted into a searcher.
///
/// When using search algorithms like [`split()`](ext/fn.split.html), users will
/// search with a `Needle` e.g. a `&str`. A needle is usually stateless,
/// however for efficient searching, we often need some preprocessing and
/// maintain a mutable state. The preprocessed structure is called the
/// [`Searcher`](trait.Searcher.html) of this needle.
///
/// The relationship between `Searcher` and `Needle` is similar to `Iterator`
/// and `IntoIterator`.
pub trait Needle<H: Haystack>: Sized
where H::Target: Hay // FIXME: RFC 2089 or 2289
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
