//! Iterators for `str` methods.

use super::pattern::{DoubleEndedSearcher, Pattern, ReverseSearcher, Searcher};
use super::validations::{next_code_point, next_code_point_reverse};
use super::{
    BytesIsNotEmpty, CharEscapeDebugContinue, CharEscapeDefault, CharEscapeUnicode,
    IsAsciiWhitespace, IsNotEmpty, IsWhitespace, LinesMap, UnsafeBytesToStr, from_utf8_unchecked,
};
use crate::fmt::{self, Write};
use crate::iter::{
    Chain, Copied, Filter, FlatMap, Flatten, FusedIterator, Map, TrustedLen, TrustedRandomAccess,
    TrustedRandomAccessNoCoerce,
};
use crate::num::NonZero;
use crate::ops::Try;
use crate::slice::{self, Split as SliceSplit};
use crate::{char as char_mod, option};

/// An iterator over the [`char`]s of a string slice.
///
///
/// This struct is created by the [`chars`] method on [`str`].
/// See its documentation for more.
///
/// [`char`]: prim@char
/// [`chars`]: str::chars
#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chars<'a> {
    pub(super) iter: slice::Iter<'a, u8>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Chars<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        // SAFETY: `str` invariant says `self.iter` is a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe { next_code_point(&mut self.iter).map(|ch| char::from_u32_unchecked(ch)) }
    }

    #[inline]
    fn count(self) -> usize {
        super::count::count_chars(self.as_str())
    }

    #[inline]
    fn advance_by(&mut self, mut remainder: usize) -> Result<(), NonZero<usize>> {
        const CHUNK_SIZE: usize = 32;

        if remainder >= CHUNK_SIZE {
            let mut chunks = self.iter.as_slice().array_chunks::<CHUNK_SIZE>();
            let mut bytes_skipped: usize = 0;

            while remainder > CHUNK_SIZE
                && let Some(chunk) = chunks.next()
            {
                bytes_skipped += CHUNK_SIZE;

                let mut start_bytes = [false; CHUNK_SIZE];

                for i in 0..CHUNK_SIZE {
                    start_bytes[i] = !super::validations::utf8_is_cont_byte(chunk[i]);
                }

                remainder -= start_bytes.into_iter().map(|i| i as u8).sum::<u8>() as usize;
            }

            // SAFETY: The amount of bytes exists since we just iterated over them,
            // so advance_by will succeed.
            unsafe { self.iter.advance_by(bytes_skipped).unwrap_unchecked() };

            // skip trailing continuation bytes
            while self.iter.len() > 0 {
                let b = self.iter.as_slice()[0];
                if !super::validations::utf8_is_cont_byte(b) {
                    break;
                }
                // SAFETY: We just peeked at the byte, therefore it exists
                unsafe { self.iter.advance_by(1).unwrap_unchecked() };
            }
        }

        while (remainder > 0) && (self.iter.len() > 0) {
            remainder -= 1;
            let b = self.iter.as_slice()[0];
            let slurp = super::validations::utf8_char_width(b);
            // SAFETY: utf8 validity requires that the string must contain
            // the continuation bytes (if any)
            unsafe { self.iter.advance_by(slurp).unwrap_unchecked() };
        }

        NonZero::new(remainder).map_or(Ok(()), Err)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        // `(len + 3)` can't overflow, because we know that the `slice::Iter`
        // belongs to a slice in memory which has a maximum length of
        // `isize::MAX` (that's well below `usize::MAX`).
        ((len + 3) / 4, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        // No need to go through the entire string.
        self.next_back()
    }
}

#[stable(feature = "chars_debug_impl", since = "1.38.0")]
impl fmt::Debug for Chars<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chars(")?;
        f.debug_list().entries(self.clone()).finish()?;
        write!(f, ")")?;
        Ok(())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Chars<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        // SAFETY: `str` invariant says `self.iter` is a valid UTF-8 string and
        // the resulting `ch` is a valid Unicode Scalar Value.
        unsafe { next_code_point_reverse(&mut self.iter).map(|ch| char::from_u32_unchecked(ch)) }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Chars<'_> {}

impl<'a> Chars<'a> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut chars = "abc".chars();
    ///
    /// assert_eq!(chars.as_str(), "abc");
    /// chars.next();
    /// assert_eq!(chars.as_str(), "bc");
    /// chars.next();
    /// chars.next();
    /// assert_eq!(chars.as_str(), "");
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &'a str {
        // SAFETY: `Chars` is only made from a str, which guarantees the iter is valid UTF-8.
        unsafe { from_utf8_unchecked(self.iter.as_slice()) }
    }
}

/// An iterator over the [`char`]s of a string slice, and their positions.
///
/// This struct is created by the [`char_indices`] method on [`str`].
/// See its documentation for more.
///
/// [`char`]: prim@char
/// [`char_indices`]: str::char_indices
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct CharIndices<'a> {
    pub(super) front_offset: usize,
    pub(super) iter: Chars<'a>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for CharIndices<'a> {
    type Item = (usize, char);

    #[inline]
    fn next(&mut self) -> Option<(usize, char)> {
        let pre_len = self.iter.iter.len();
        match self.iter.next() {
            None => None,
            Some(ch) => {
                let index = self.front_offset;
                let len = self.iter.iter.len();
                self.front_offset += pre_len - len;
                Some((index, ch))
            }
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<(usize, char)> {
        // No need to go through the entire string.
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for CharIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, char)> {
        self.iter.next_back().map(|ch| {
            let index = self.front_offset + self.iter.iter.len();
            (index, ch)
        })
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for CharIndices<'_> {}

impl<'a> CharIndices<'a> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    #[must_use]
    #[inline]
    pub fn as_str(&self) -> &'a str {
        self.iter.as_str()
    }

    /// Returns the byte position of the next character, or the length
    /// of the underlying string if there are no more characters.
    ///
    /// This means that, when the iterator has not been fully consumed,
    /// the returned value will match the index that will be returned
    /// by the next call to [`next()`](Self::next).
    ///
    /// # Examples
    ///
    /// ```
    /// let mut chars = "a楽".char_indices();
    ///
    /// // `next()` has not been called yet, so `offset()` returns the byte
    /// // index of the first character of the string, which is always 0.
    /// assert_eq!(chars.offset(), 0);
    /// // As expected, the first call to `next()` also returns 0 as index.
    /// assert_eq!(chars.next(), Some((0, 'a')));
    ///
    /// // `next()` has been called once, so `offset()` returns the byte index
    /// // of the second character ...
    /// assert_eq!(chars.offset(), 1);
    /// // ... which matches the index returned by the next call to `next()`.
    /// assert_eq!(chars.next(), Some((1, '楽')));
    ///
    /// // Once the iterator has been consumed, `offset()` returns the length
    /// // in bytes of the string.
    /// assert_eq!(chars.offset(), 4);
    /// assert_eq!(chars.next(), None);
    /// ```
    #[inline]
    #[must_use]
    #[stable(feature = "char_indices_offset", since = "1.82.0")]
    pub fn offset(&self) -> usize {
        self.front_offset
    }
}

/// An iterator over the bytes of a string slice.
///
/// This struct is created by the [`bytes`] method on [`str`].
/// See its documentation for more.
///
/// [`bytes`]: str::bytes
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct Bytes<'a>(pub(super) Copied<slice::Iter<'a, u8>>);

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for Bytes<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }

    #[inline]
    fn all<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.0.all(f)
    }

    #[inline]
    fn any<F>(&mut self, f: F) -> bool
    where
        F: FnMut(Self::Item) -> bool,
    {
        self.0.any(f)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.0.find(predicate)
    }

    #[inline]
    fn position<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        self.0.position(predicate)
    }

    #[inline]
    fn rposition<P>(&mut self, predicate: P) -> Option<usize>
    where
        P: FnMut(Self::Item) -> bool,
    {
        self.0.rposition(predicate)
    }

    #[inline]
    unsafe fn __iterator_get_unchecked(&mut self, idx: usize) -> u8 {
        // SAFETY: the caller must uphold the safety contract
        // for `Iterator::__iterator_get_unchecked`.
        unsafe { self.0.__iterator_get_unchecked(idx) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl DoubleEndedIterator for Bytes<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<u8> {
        self.0.next_back()
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth_back(n)
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        self.0.rfind(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ExactSizeIterator for Bytes<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Bytes<'_> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl TrustedLen for Bytes<'_> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl TrustedRandomAccess for Bytes<'_> {}

#[doc(hidden)]
#[unstable(feature = "trusted_random_access", issue = "none")]
unsafe impl TrustedRandomAccessNoCoerce for Bytes<'_> {
    const MAY_HAVE_SIDE_EFFECT: bool = false;
}

/// This macro generates a Clone impl for string pattern API
/// wrapper types of the form X<'a, P>
macro_rules! derive_pattern_clone {
    (clone $t:ident with |$s:ident| $e:expr) => {
        impl<'a, P> Clone for $t<'a, P>
        where
            P: Pattern<Searcher<'a>: Clone>,
        {
            fn clone(&self) -> Self {
                let $s = self;
                $e
            }
        }
    };
}

/// This macro generates two public iterator structs
/// wrapping a private internal one that makes use of the `Pattern` API.
///
/// For all patterns `P: Pattern` the following items will be
/// generated (generics omitted):
///
/// struct $forward_iterator($internal_iterator);
/// struct $reverse_iterator($internal_iterator);
///
/// impl Iterator for $forward_iterator
/// { /* internal ends up calling Searcher::next_match() */ }
///
/// impl DoubleEndedIterator for $forward_iterator
///       where P::Searcher: DoubleEndedSearcher
/// { /* internal ends up calling Searcher::next_match_back() */ }
///
/// impl Iterator for $reverse_iterator
///       where P::Searcher: ReverseSearcher
/// { /* internal ends up calling Searcher::next_match_back() */ }
///
/// impl DoubleEndedIterator for $reverse_iterator
///       where P::Searcher: DoubleEndedSearcher
/// { /* internal ends up calling Searcher::next_match() */ }
///
/// The internal one is defined outside the macro, and has almost the same
/// semantic as a DoubleEndedIterator by delegating to `pattern::Searcher` and
/// `pattern::ReverseSearcher` for both forward and reverse iteration.
///
/// "Almost", because a `Searcher` and a `ReverseSearcher` for a given
/// `Pattern` might not return the same elements, so actually implementing
/// `DoubleEndedIterator` for it would be incorrect.
/// (See the docs in `str::pattern` for more details)
///
/// However, the internal struct still represents a single ended iterator from
/// either end, and depending on pattern is also a valid double ended iterator,
/// so the two wrapper structs implement `Iterator`
/// and `DoubleEndedIterator` depending on the concrete pattern type, leading
/// to the complex impls seen above.
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
        pub struct $forward_iterator<'a, P: Pattern>(pub(super) $internal_iterator<'a, P>);

        $(#[$common_stability_attribute])*
        impl<'a, P> fmt::Debug for $forward_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: fmt::Debug>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($forward_iterator))
                    .field(&self.0)
                    .finish()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern> Iterator for $forward_iterator<'a, P> {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<$iterty> {
                self.0.next()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P> Clone for $forward_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: Clone>,
        {
            fn clone(&self) -> Self {
                $forward_iterator(self.0.clone())
            }
        }

        $(#[$reverse_iterator_attribute])*
        $(#[$common_stability_attribute])*
        pub struct $reverse_iterator<'a, P: Pattern>(pub(super) $internal_iterator<'a, P>);

        $(#[$common_stability_attribute])*
        impl<'a, P> fmt::Debug for $reverse_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: fmt::Debug>,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($reverse_iterator))
                    .field(&self.0)
                    .finish()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P> Iterator for $reverse_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: ReverseSearcher<'a>>,
        {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<$iterty> {
                self.0.next_back()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P> Clone for $reverse_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: Clone>,
        {
            fn clone(&self) -> Self {
                $reverse_iterator(self.0.clone())
            }
        }

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, P: Pattern> FusedIterator for $forward_iterator<'a, P> {}

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, P> FusedIterator for $reverse_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: ReverseSearcher<'a>>,
        {}

        generate_pattern_iterators!($($t)* with $(#[$common_stability_attribute])*,
                                                $forward_iterator,
                                                $reverse_iterator, $iterty);
    };
    {
        double ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
    } => {
        $(#[$common_stability_attribute])*
        impl<'a, P> DoubleEndedIterator for $forward_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: DoubleEndedSearcher<'a>>,
        {
            #[inline]
            fn next_back(&mut self) -> Option<$iterty> {
                self.0.next_back()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P> DoubleEndedIterator for $reverse_iterator<'a, P>
        where
            P: Pattern<Searcher<'a>: DoubleEndedSearcher<'a>>,
        {
            #[inline]
            fn next_back(&mut self) -> Option<$iterty> {
                self.0.next()
            }
        }
    };
    {
        single ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
    } => {}
}

derive_pattern_clone! {
    clone SplitInternal
    with |s| SplitInternal { matcher: s.matcher.clone(), ..*s }
}

pub(super) struct SplitInternal<'a, P: Pattern> {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) matcher: P::Searcher<'a>,
    pub(super) allow_trailing_empty: bool,
    pub(super) finished: bool,
}

impl<'a, P> fmt::Debug for SplitInternal<'a, P>
where
    P: Pattern<Searcher<'a>: fmt::Debug>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitInternal")
            .field("start", &self.start)
            .field("end", &self.end)
            .field("matcher", &self.matcher)
            .field("allow_trailing_empty", &self.allow_trailing_empty)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, P: Pattern> SplitInternal<'a, P> {
    #[inline]
    fn get_end(&mut self) -> Option<&'a str> {
        if !self.finished {
            self.finished = true;

            if self.allow_trailing_empty || self.end - self.start > 0 {
                // SAFETY: `self.start` and `self.end` always lie on unicode boundaries.
                let string = unsafe { self.matcher.haystack().get_unchecked(self.start..self.end) };
                return Some(string);
            }
        }

        None
    }

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.finished {
            return None;
        }

        let haystack = self.matcher.haystack();
        match self.matcher.next_match() {
            // SAFETY: `Searcher` guarantees that `a` and `b` lie on unicode boundaries.
            Some((a, b)) => unsafe {
                let elt = haystack.get_unchecked(self.start..a);
                self.start = b;
                Some(elt)
            },
            None => self.get_end(),
        }
    }

    #[inline]
    fn next_inclusive(&mut self) -> Option<&'a str> {
        if self.finished {
            return None;
        }

        let haystack = self.matcher.haystack();
        match self.matcher.next_match() {
            // SAFETY: `Searcher` guarantees that `b` lies on unicode boundary,
            // and self.start is either the start of the original string,
            // or `b` was assigned to it, so it also lies on unicode boundary.
            Some((_, b)) => unsafe {
                let elt = haystack.get_unchecked(self.start..b);
                self.start = b;
                Some(elt)
            },
            None => self.get_end(),
        }
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
    where
        P::Searcher<'a>: ReverseSearcher<'a>,
    {
        if self.finished {
            return None;
        }

        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.next_back() {
                Some(elt) if !elt.is_empty() => return Some(elt),
                _ => {
                    if self.finished {
                        return None;
                    }
                }
            }
        }

        let haystack = self.matcher.haystack();
        match self.matcher.next_match_back() {
            // SAFETY: `Searcher` guarantees that `a` and `b` lie on unicode boundaries.
            Some((a, b)) => unsafe {
                let elt = haystack.get_unchecked(b..self.end);
                self.end = a;
                Some(elt)
            },
            // SAFETY: `self.start` and `self.end` always lie on unicode boundaries.
            None => unsafe {
                self.finished = true;
                Some(haystack.get_unchecked(self.start..self.end))
            },
        }
    }

    #[inline]
    fn next_back_inclusive(&mut self) -> Option<&'a str>
    where
        P::Searcher<'a>: ReverseSearcher<'a>,
    {
        if self.finished {
            return None;
        }

        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.next_back_inclusive() {
                Some(elt) if !elt.is_empty() => return Some(elt),
                _ => {
                    if self.finished {
                        return None;
                    }
                }
            }
        }

        let haystack = self.matcher.haystack();
        match self.matcher.next_match_back() {
            // SAFETY: `Searcher` guarantees that `b` lies on unicode boundary,
            // and self.end is either the end of the original string,
            // or `b` was assigned to it, so it also lies on unicode boundary.
            Some((_, b)) => unsafe {
                let elt = haystack.get_unchecked(b..self.end);
                self.end = b;
                Some(elt)
            },
            // SAFETY: self.start is either the start of the original string,
            // or start of a substring that represents the part of the string that hasn't
            // iterated yet. Either way, it is guaranteed to lie on unicode boundary.
            // self.end is either the end of the original string,
            // or `b` was assigned to it, so it also lies on unicode boundary.
            None => unsafe {
                self.finished = true;
                Some(haystack.get_unchecked(self.start..self.end))
            },
        }
    }

    #[inline]
    fn remainder(&self) -> Option<&'a str> {
        // `Self::get_end` doesn't change `self.start`
        if self.finished {
            return None;
        }

        // SAFETY: `self.start` and `self.end` always lie on unicode boundaries.
        Some(unsafe { self.matcher.haystack().get_unchecked(self.start..self.end) })
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`split`].
        ///
        /// [`split`]: str::split
        struct Split;
    reverse:
        /// Created with the method [`rsplit`].
        ///
        /// [`rsplit`]: str::rsplit
        struct RSplit;
    stability:
        #[stable(feature = "rust1", since = "1.0.0")]
    internal:
        SplitInternal yielding (&'a str);
    delegate double ended;
}

impl<'a, P: Pattern> Split<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_remainder)]
    /// let mut split = "Mary had a little lamb".split(' ');
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    /// split.next();
    /// assert_eq!(split.remainder(), Some("had a little lamb"));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

impl<'a, P: Pattern> RSplit<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_remainder)]
    /// let mut split = "Mary had a little lamb".rsplit(' ');
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    /// split.next();
    /// assert_eq!(split.remainder(), Some("Mary had a little"));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`split_terminator`].
        ///
        /// [`split_terminator`]: str::split_terminator
        struct SplitTerminator;
    reverse:
        /// Created with the method [`rsplit_terminator`].
        ///
        /// [`rsplit_terminator`]: str::rsplit_terminator
        struct RSplitTerminator;
    stability:
        #[stable(feature = "rust1", since = "1.0.0")]
    internal:
        SplitInternal yielding (&'a str);
    delegate double ended;
}

impl<'a, P: Pattern> SplitTerminator<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_remainder)]
    /// let mut split = "A..B..".split_terminator('.');
    /// assert_eq!(split.remainder(), Some("A..B.."));
    /// split.next();
    /// assert_eq!(split.remainder(), Some(".B.."));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

impl<'a, P: Pattern> RSplitTerminator<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_remainder)]
    /// let mut split = "A..B..".rsplit_terminator('.');
    /// assert_eq!(split.remainder(), Some("A..B.."));
    /// split.next();
    /// assert_eq!(split.remainder(), Some("A..B"));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

derive_pattern_clone! {
    clone SplitNInternal
    with |s| SplitNInternal { iter: s.iter.clone(), ..*s }
}

pub(super) struct SplitNInternal<'a, P: Pattern> {
    pub(super) iter: SplitInternal<'a, P>,
    /// The number of splits remaining
    pub(super) count: usize,
}

impl<'a, P> fmt::Debug for SplitNInternal<'a, P>
where
    P: Pattern<Searcher<'a>: fmt::Debug>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitNInternal")
            .field("iter", &self.iter)
            .field("count", &self.count)
            .finish()
    }
}

impl<'a, P: Pattern> SplitNInternal<'a, P> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        match self.count {
            0 => None,
            1 => {
                self.count = 0;
                self.iter.get_end()
            }
            _ => {
                self.count -= 1;
                self.iter.next()
            }
        }
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
    where
        P::Searcher<'a>: ReverseSearcher<'a>,
    {
        match self.count {
            0 => None,
            1 => {
                self.count = 0;
                self.iter.get_end()
            }
            _ => {
                self.count -= 1;
                self.iter.next_back()
            }
        }
    }

    #[inline]
    fn remainder(&self) -> Option<&'a str> {
        self.iter.remainder()
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`splitn`].
        ///
        /// [`splitn`]: str::splitn
        struct SplitN;
    reverse:
        /// Created with the method [`rsplitn`].
        ///
        /// [`rsplitn`]: str::rsplitn
        struct RSplitN;
    stability:
        #[stable(feature = "rust1", since = "1.0.0")]
    internal:
        SplitNInternal yielding (&'a str);
    delegate single ended;
}

impl<'a, P: Pattern> SplitN<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_remainder)]
    /// let mut split = "Mary had a little lamb".splitn(3, ' ');
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    /// split.next();
    /// assert_eq!(split.remainder(), Some("had a little lamb"));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

impl<'a, P: Pattern> RSplitN<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_remainder)]
    /// let mut split = "Mary had a little lamb".rsplitn(3, ' ');
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    /// split.next();
    /// assert_eq!(split.remainder(), Some("Mary had a little"));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

derive_pattern_clone! {
    clone MatchIndicesInternal
    with |s| MatchIndicesInternal(s.0.clone())
}

pub(super) struct MatchIndicesInternal<'a, P: Pattern>(pub(super) P::Searcher<'a>);

impl<'a, P> fmt::Debug for MatchIndicesInternal<'a, P>
where
    P: Pattern<Searcher<'a>: fmt::Debug>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MatchIndicesInternal").field(&self.0).finish()
    }
}

impl<'a, P: Pattern> MatchIndicesInternal<'a, P> {
    #[inline]
    fn next(&mut self) -> Option<(usize, &'a str)> {
        self.0
            .next_match()
            // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
            .map(|(start, end)| unsafe { (start, self.0.haystack().get_unchecked(start..end)) })
    }

    #[inline]
    fn next_back(&mut self) -> Option<(usize, &'a str)>
    where
        P::Searcher<'a>: ReverseSearcher<'a>,
    {
        self.0
            .next_match_back()
            // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
            .map(|(start, end)| unsafe { (start, self.0.haystack().get_unchecked(start..end)) })
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`match_indices`].
        ///
        /// [`match_indices`]: str::match_indices
        struct MatchIndices;
    reverse:
        /// Created with the method [`rmatch_indices`].
        ///
        /// [`rmatch_indices`]: str::rmatch_indices
        struct RMatchIndices;
    stability:
        #[stable(feature = "str_match_indices", since = "1.5.0")]
    internal:
        MatchIndicesInternal yielding ((usize, &'a str));
    delegate double ended;
}

derive_pattern_clone! {
    clone MatchesInternal
    with |s| MatchesInternal(s.0.clone())
}

pub(super) struct MatchesInternal<'a, P: Pattern>(pub(super) P::Searcher<'a>);

impl<'a, P> fmt::Debug for MatchesInternal<'a, P>
where
    P: Pattern<Searcher<'a>: fmt::Debug>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MatchesInternal").field(&self.0).finish()
    }
}

impl<'a, P: Pattern> MatchesInternal<'a, P> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
        self.0.next_match().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().get_unchecked(a..b)
        })
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
    where
        P::Searcher<'a>: ReverseSearcher<'a>,
    {
        // SAFETY: `Searcher` guarantees that `start` and `end` lie on unicode boundaries.
        self.0.next_match_back().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().get_unchecked(a..b)
        })
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`matches`].
        ///
        /// [`matches`]: str::matches
        struct Matches;
    reverse:
        /// Created with the method [`rmatches`].
        ///
        /// [`rmatches`]: str::rmatches
        struct RMatches;
    stability:
        #[stable(feature = "str_matches", since = "1.2.0")]
    internal:
        MatchesInternal yielding (&'a str);
    delegate double ended;
}

/// An iterator over the lines of a string, as string slices.
///
/// This struct is created with the [`lines`] method on [`str`].
/// See its documentation for more.
///
/// [`lines`]: str::lines
#[stable(feature = "rust1", since = "1.0.0")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct Lines<'a>(pub(super) Map<SplitInclusive<'a, char>, LinesMap>);

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Lines<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<&'a str> {
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Lines<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.0.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Lines<'_> {}

impl<'a> Lines<'a> {
    /// Returns the remaining lines of the split string.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_lines_remainder)]
    ///
    /// let mut lines = "a\nb\nc\nd".lines();
    /// assert_eq!(lines.remainder(), Some("a\nb\nc\nd"));
    ///
    /// lines.next();
    /// assert_eq!(lines.remainder(), Some("b\nc\nd"));
    ///
    /// lines.by_ref().for_each(drop);
    /// assert_eq!(lines.remainder(), None);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "str_lines_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.iter.remainder()
    }
}

/// Created with the method [`lines_any`].
///
/// [`lines_any`]: str::lines_any
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "1.4.0", note = "use lines()/Lines instead now")]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
#[allow(deprecated)]
pub struct LinesAny<'a>(pub(super) Lines<'a>);

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl<'a> Iterator for LinesAny<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl<'a> DoubleEndedIterator for LinesAny<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.0.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
#[allow(deprecated)]
impl FusedIterator for LinesAny<'_> {}

/// An iterator over the non-whitespace substrings of a string,
/// separated by any amount of whitespace.
///
/// This struct is created by the [`split_whitespace`] method on [`str`].
/// See its documentation for more.
///
/// [`split_whitespace`]: str::split_whitespace
#[stable(feature = "split_whitespace", since = "1.1.0")]
#[derive(Clone, Debug)]
pub struct SplitWhitespace<'a> {
    pub(super) inner: Filter<Split<'a, IsWhitespace>, IsNotEmpty>,
}

/// An iterator over the non-ASCII-whitespace substrings of a string,
/// separated by any amount of ASCII whitespace.
///
/// This struct is created by the [`split_ascii_whitespace`] method on [`str`].
/// See its documentation for more.
///
/// [`split_ascii_whitespace`]: str::split_ascii_whitespace
#[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
#[derive(Clone, Debug)]
pub struct SplitAsciiWhitespace<'a> {
    pub(super) inner:
        Map<Filter<SliceSplit<'a, u8, IsAsciiWhitespace>, BytesIsNotEmpty>, UnsafeBytesToStr>,
}

/// An iterator over the substrings of a string,
/// terminated by a substring matching to a predicate function
/// Unlike `Split`, it contains the matched part as a terminator
/// of the subslice.
///
/// This struct is created by the [`split_inclusive`] method on [`str`].
/// See its documentation for more.
///
/// [`split_inclusive`]: str::split_inclusive
#[stable(feature = "split_inclusive", since = "1.51.0")]
pub struct SplitInclusive<'a, P: Pattern>(pub(super) SplitInternal<'a, P>);

#[stable(feature = "split_whitespace", since = "1.1.0")]
impl<'a> Iterator for SplitWhitespace<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<&'a str> {
        self.next_back()
    }
}

#[stable(feature = "split_whitespace", since = "1.1.0")]
impl<'a> DoubleEndedIterator for SplitWhitespace<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.inner.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for SplitWhitespace<'_> {}

impl<'a> SplitWhitespace<'a> {
    /// Returns remainder of the split string
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_whitespace_remainder)]
    ///
    /// let mut split = "Mary had a little lamb".split_whitespace();
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    ///
    /// split.next();
    /// assert_eq!(split.remainder(), Some("had a little lamb"));
    ///
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "str_split_whitespace_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.inner.iter.remainder()
    }
}

#[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
impl<'a> Iterator for SplitAsciiWhitespace<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<&'a str> {
        self.next_back()
    }
}

#[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
impl<'a> DoubleEndedIterator for SplitAsciiWhitespace<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.inner.next_back()
    }
}

#[stable(feature = "split_ascii_whitespace", since = "1.34.0")]
impl FusedIterator for SplitAsciiWhitespace<'_> {}

impl<'a> SplitAsciiWhitespace<'a> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_whitespace_remainder)]
    ///
    /// let mut split = "Mary had a little lamb".split_ascii_whitespace();
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    ///
    /// split.next();
    /// assert_eq!(split.remainder(), Some("had a little lamb"));
    ///
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "str_split_whitespace_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        if self.inner.iter.iter.finished {
            return None;
        }

        // SAFETY: Slice is created from str.
        Some(unsafe { crate::str::from_utf8_unchecked(&self.inner.iter.iter.v) })
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, P: Pattern> Iterator for SplitInclusive<'a, P> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0.next_inclusive()
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, P: Pattern<Searcher<'a>: fmt::Debug>> fmt::Debug for SplitInclusive<'a, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SplitInclusive").field("0", &self.0).finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, P: Pattern<Searcher<'a>: Clone>> Clone for SplitInclusive<'a, P> {
    fn clone(&self) -> Self {
        SplitInclusive(self.0.clone())
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, P: Pattern<Searcher<'a>: DoubleEndedSearcher<'a>>> DoubleEndedIterator
    for SplitInclusive<'a, P>
{
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.0.next_back_inclusive()
    }
}

#[stable(feature = "split_inclusive", since = "1.51.0")]
impl<'a, P: Pattern> FusedIterator for SplitInclusive<'a, P> {}

impl<'a, P: Pattern> SplitInclusive<'a, P> {
    /// Returns remainder of the split string.
    ///
    /// If the iterator is empty, returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(str_split_inclusive_remainder)]
    /// let mut split = "Mary had a little lamb".split_inclusive(' ');
    /// assert_eq!(split.remainder(), Some("Mary had a little lamb"));
    /// split.next();
    /// assert_eq!(split.remainder(), Some("had a little lamb"));
    /// split.by_ref().for_each(drop);
    /// assert_eq!(split.remainder(), None);
    /// ```
    #[inline]
    #[unstable(feature = "str_split_inclusive_remainder", issue = "77998")]
    pub fn remainder(&self) -> Option<&'a str> {
        self.0.remainder()
    }
}

/// An iterator of [`u16`] over the string encoded as UTF-16.
///
/// This struct is created by the [`encode_utf16`] method on [`str`].
/// See its documentation for more.
///
/// [`encode_utf16`]: str::encode_utf16
#[derive(Clone)]
#[stable(feature = "encode_utf16", since = "1.8.0")]
pub struct EncodeUtf16<'a> {
    pub(super) chars: Chars<'a>,
    pub(super) extra: u16,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl fmt::Debug for EncodeUtf16<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodeUtf16").finish_non_exhaustive()
    }
}

#[stable(feature = "encode_utf16", since = "1.8.0")]
impl<'a> Iterator for EncodeUtf16<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0; 2];
        self.chars.next().map(|ch| {
            let n = ch.encode_utf16(&mut buf).len();
            if n == 2 {
                self.extra = buf[1];
            }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.chars.iter.len();
        // The highest bytes:code units ratio occurs for 3-byte sequences,
        // since a 4-byte sequence results in 2 code units. The lower bound
        // is therefore determined by assuming the remaining bytes contain as
        // many 3-byte sequences as possible. The highest bytes:code units
        // ratio is for 1-byte sequences, so use this for the upper bound.
        // `(len + 2)` can't overflow, because we know that the `slice::Iter`
        // belongs to a slice in memory which has a maximum length of
        // `isize::MAX` (that's well below `usize::MAX`)
        if self.extra == 0 {
            ((len + 2) / 3, Some(len))
        } else {
            // We're in the middle of a surrogate pair, so add the remaining
            // surrogate to the bounds.
            ((len + 2) / 3 + 1, Some(len + 1))
        }
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for EncodeUtf16<'_> {}

/// The return type of [`str::escape_debug`].
#[stable(feature = "str_escape", since = "1.34.0")]
#[derive(Clone, Debug)]
pub struct EscapeDebug<'a> {
    pub(super) inner: Chain<
        Flatten<option::IntoIter<char_mod::EscapeDebug>>,
        FlatMap<Chars<'a>, char_mod::EscapeDebug, CharEscapeDebugContinue>,
    >,
}

/// The return type of [`str::escape_default`].
#[stable(feature = "str_escape", since = "1.34.0")]
#[derive(Clone, Debug)]
pub struct EscapeDefault<'a> {
    pub(super) inner: FlatMap<Chars<'a>, char_mod::EscapeDefault, CharEscapeDefault>,
}

/// The return type of [`str::escape_unicode`].
#[stable(feature = "str_escape", since = "1.34.0")]
#[derive(Clone, Debug)]
pub struct EscapeUnicode<'a> {
    pub(super) inner: FlatMap<Chars<'a>, char_mod::EscapeUnicode, CharEscapeUnicode>,
}

macro_rules! escape_types_impls {
    ($( $Name: ident ),+) => {$(
        #[stable(feature = "str_escape", since = "1.34.0")]
        impl<'a> fmt::Display for $Name<'a> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.clone().try_for_each(|c| f.write_char(c))
            }
        }

        #[stable(feature = "str_escape", since = "1.34.0")]
        impl<'a> Iterator for $Name<'a> {
            type Item = char;

            #[inline]
            fn next(&mut self) -> Option<char> { self.inner.next() }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }

            #[inline]
            fn try_fold<Acc, Fold, R>(&mut self, init: Acc, fold: Fold) -> R where
                Self: Sized, Fold: FnMut(Acc, Self::Item) -> R, R: Try<Output = Acc>
            {
                self.inner.try_fold(init, fold)
            }

            #[inline]
            fn fold<Acc, Fold>(self, init: Acc, fold: Fold) -> Acc
                where Fold: FnMut(Acc, Self::Item) -> Acc,
            {
                self.inner.fold(init, fold)
            }
        }

        #[stable(feature = "str_escape", since = "1.34.0")]
        impl<'a> FusedIterator for $Name<'a> {}
    )+}
}

escape_types_impls!(EscapeDebug, EscapeDefault, EscapeUnicode);
