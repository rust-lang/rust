use crate::needle::*;
use crate::ops::Range;
use crate::cmp::{Ordering, max, min};
use crate::usize;
use crate::fmt;

//------------------------------------------------------------------------------
// Element searcher
//------------------------------------------------------------------------------

#[derive(Clone)]
pub struct ElemSearcher<F> {
    predicate: F,
}

// we need to impl Debug for everything due to stability guarantee.
impl<F> fmt::Debug for ElemSearcher<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ElemSearcher").finish()
    }
}

macro_rules! impl_needle_with_elem_searcher {
    (<[$($gen:tt)*]> $ty:ty) => {
        impl<$($gen)*> Needle<$ty> for F
        where
            F: FnMut(&T) -> bool,
        {
            type Searcher = ElemSearcher<F>;
            type Consumer = ElemSearcher<F>;

            #[inline]
            fn into_searcher(self) -> Self::Searcher {
                ElemSearcher {
                    predicate: self,
                }
            }

            #[inline]
            fn into_consumer(self) -> Self::Consumer {
                ElemSearcher {
                    predicate: self,
                }
            }
        }
    }
}

impl_needle_with_elem_searcher!(<['h, T, F]> &'h [T]);
impl_needle_with_elem_searcher!(<['h, T, F]> &'h mut [T]);

unsafe impl<T, F> Searcher<[T]> for ElemSearcher<F>
where
    F: FnMut(&T) -> bool,
{
    #[inline]
    fn search(&mut self, span: Span<&[T]>) -> Option<Range<usize>> {
        let (rest, range) = span.into_parts();
        let start = range.start;
        let pos = rest[range].iter().position(&mut self.predicate)?;
        Some((pos + start)..(pos + start + 1))
    }
}

unsafe impl<T, F> Consumer<[T]> for ElemSearcher<F>
where
    F: FnMut(&T) -> bool,
{
    #[inline]
    fn consume(&mut self, span: Span<&[T]>) -> Option<usize> {
        let (hay, range) = span.into_parts();
        if range.end == range.start {
            return None;
        }
        let x = unsafe { hay.get_unchecked(range.start) };
        if (self.predicate)(x) {
            Some(range.start + 1)
        } else {
            None
        }
    }

    #[inline]
    fn trim_start(&mut self, hay: &[T]) -> usize {
        let mut it = hay.iter();
        let len = hay.len();
        if it.find(|x| !(self.predicate)(x)).is_some() {
            len - it.as_slice().len() - 1
        } else {
            len
        }
    }
}

unsafe impl<T, F> ReverseSearcher<[T]> for ElemSearcher<F>
where
    F: FnMut(&T) -> bool,
{
    #[inline]
    fn rsearch(&mut self, span: Span<&[T]>) -> Option<Range<usize>> {
        let (rest, range) = span.into_parts();
        let start = range.start;
        let pos = rest[range].iter().rposition(&mut self.predicate)?;
        Some((pos + start)..(pos + start + 1))
    }
}

unsafe impl<T, F> ReverseConsumer<[T]> for ElemSearcher<F>
where
    F: FnMut(&T) -> bool,
{
    #[inline]
    fn rconsume(&mut self, span: Span<&[T]>) -> Option<usize> {
        let (hay, range) = span.into_parts();
        if range.start == range.end {
            return None;
        }
        let last = range.end - 1;
        let x = unsafe { hay.get_unchecked(last) };
        if (self.predicate)(x) {
            Some(last)
        } else {
            None
        }
    }

    #[inline]
    fn trim_end(&mut self, hay: &[T]) -> usize {
        hay.iter().rposition(|x| !(self.predicate)(x)).map_or(0, |p| p + 1)
    }
}

unsafe impl<T, F> DoubleEndedSearcher<[T]> for ElemSearcher<F>
where
    F: FnMut(&T) -> bool,
{}

unsafe impl<T, F> DoubleEndedConsumer<[T]> for ElemSearcher<F>
where
    F: FnMut(&T) -> bool,
{}

//------------------------------------------------------------------------------
// Two way searcher helpers
//------------------------------------------------------------------------------

type FastSkipByteset = u64;

trait FastSkipOptimization {
    fn byteset_mask(&self) -> FastSkipByteset;
}

impl<T: ?Sized> FastSkipOptimization for T {
    #[inline]
    default fn byteset_mask(&self) -> FastSkipByteset { !0 }
}

impl FastSkipOptimization for u8 {
    #[inline]
    fn byteset_mask(&self) -> FastSkipByteset { 1 << (self & 63) }
}

trait MaximalSuffix: Sized {
    // Compute the maximal suffix of `&[T]`.
    //
    // The maximal suffix is a possible critical factorization (u, v) of `arr`.
    //
    // Returns (`i`, `p`) where `i` is the starting index of v and `p` is the
    // period of v.
    //
    // `order` determines if lexical order is `<` or `>`. Both
    // orders must be computed -- the ordering with the largest `i` gives
    // a critical factorization.
    //
    // For long period cases, the resulting period is not exact (it is too short).
    fn maximal_suffix(arr: &[Self], order: Ordering) -> (usize, usize);

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
    fn reverse_maximal_suffix(arr: &[Self], known_period: usize, order: Ordering) -> usize;
}

// fallback to naive search for non-Ord slices.
impl<T: PartialEq> MaximalSuffix for T {
    default fn maximal_suffix(_: &[Self], _: Ordering) -> (usize, usize) {
        (0, 1)
    }

    default fn reverse_maximal_suffix(_: &[Self], _: usize, _: Ordering) -> usize {
        0
    }
}

impl<T: Ord> MaximalSuffix for T {
    fn maximal_suffix(arr: &[Self], order: Ordering) -> (usize, usize) {
        let mut left = 0; // Corresponds to i in the paper
        let mut right = 1; // Corresponds to j in the paper
        let mut offset = 0; // Corresponds to k in the paper, but starting at 0
                            // to match 0-based indexing.
        let mut period = 1; // Corresponds to p in the paper

        while let Some(a) = arr.get(right + offset) {
            // `left` will be inbounds when `right` is.
            let b = &arr[left + offset];
            match a.cmp(b) {
                Ordering::Equal => {
                    // Advance through repetition of the current period.
                    if offset + 1 == period {
                        right += offset + 1;
                        offset = 0;
                    } else {
                        offset += 1;
                    }
                }
                o if o == order => {
                    // Suffix is smaller, period is entire prefix so far.
                    right += offset + 1;
                    offset = 0;
                    period = right - left;
                }
                _ => {
                    // Suffix is larger, start over from current location.
                    left = right;
                    right += 1;
                    offset = 0;
                    period = 1;
                }
            };
        }
        (left, period)
    }

    fn reverse_maximal_suffix(arr: &[Self], known_period: usize, order: Ordering) -> usize {
        let mut left = 0; // Corresponds to i in the paper
        let mut right = 1; // Corresponds to j in the paper
        let mut offset = 0; // Corresponds to k in the paper, but starting at 0
                            // to match 0-based indexing.
        let mut period = 1; // Corresponds to p in the paper
        let n = arr.len();

        while right + offset < n {
            let a = &arr[n - (1 + right + offset)];
            let b = &arr[n - (1 + left + offset)];
            match a.cmp(b) {
                Ordering::Equal => {
                    // Advance through repetition of the current period.
                    if offset + 1 == period {
                        right += offset + 1;
                        offset = 0;
                    } else {
                        offset += 1;
                    }
                }
                o if o == order => {
                    // Suffix is smaller, period is entire prefix so far.
                    right += offset + 1;
                    offset = 0;
                    period = right - left;
                }
                _ => {
                    // Suffix is larger, start over from current location.
                    left = right;
                    right += 1;
                    offset = 0;
                    period = 1;
                }
            }
            if period == known_period {
                break;
            }
        }
        debug_assert!(period <= known_period);
        left
    }
}

//------------------------------------------------------------------------------
// Two way searcher
//------------------------------------------------------------------------------

struct LongPeriod;
struct ShortPeriod;

trait Period {
    const IS_LONG_PERIOD: bool;
}
impl Period for LongPeriod {
    const IS_LONG_PERIOD: bool = true;
}
impl Period for ShortPeriod {
    const IS_LONG_PERIOD: bool = false;
}

/// A slice searcher based on Two-Way algorithm.
#[derive(Debug)]
pub struct TwoWaySearcher<'p, T: 'p> {
    // constants
    /// critical factorization index
    crit_pos: usize,
    /// critical factorization index for reversed needle
    crit_pos_back: usize,

    period: usize,

    /// `byteset` is an extension (not part of the two way algorithm);
    /// it's a 64-bit "fingerprint" where each set bit `j` corresponds
    /// to a (byte & 63) == j present in the needle.
    byteset: FastSkipByteset,

    needle: &'p [T],

    // variables
    /// index into needle before which we have already matched
    memory: usize,
    /// index into needle after which we have already matched
    memory_back: usize,
}

impl<'p, T: 'p> Clone for TwoWaySearcher<'p, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'p, T: 'p> Copy for TwoWaySearcher<'p, T> {}

impl<'p, T> TwoWaySearcher<'p, T>
where
    T: PartialEq + 'p,
{
    #[inline]
    fn do_next<P: Period>(&mut self, hay: &[T], range: Range<usize>) -> Option<Range<usize>> {
        let needle = self.needle;

        let mut position = range.start;
        'search: loop {
            // Check that we have room to search in
            // position + needle_last can not overflow if we assume slices
            // are bounded by isize's range.
            let i = position + (needle.len() - 1);
            if i >= range.end {
                return None;
            }
            // let tail_item = &hay[i]; // using get_unchecked here would be slower
            let tail_item = unsafe { hay.get_unchecked(i) };

            // Quickly skip by large portions unrelated to our substring
            if !self.byteset_contains(tail_item) {
                position += needle.len();
                if !P::IS_LONG_PERIOD {
                    self.memory = 0;
                }
                continue 'search;
            }

            // See if the right part of the needle matches
            let start = if P::IS_LONG_PERIOD {
                self.crit_pos
            } else {
                max(self.crit_pos, self.memory)
            };
            for i in start..needle.len() {
                if unsafe { needle.get_unchecked(i) != hay.get_unchecked(position + i) } {
                    position += i - self.crit_pos + 1;
                    if !P::IS_LONG_PERIOD {
                        self.memory = 0;
                    }
                    continue 'search;
                }
            }

            // See if the left part of the needle matches
            let start = if P::IS_LONG_PERIOD { 0 } else { self.memory };
            for i in (start..self.crit_pos).rev() {
                if unsafe { needle.get_unchecked(i) != hay.get_unchecked(position + i) } {
                    position += self.period;
                    if !P::IS_LONG_PERIOD {
                        self.memory = needle.len() - self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            // Note: add self.period instead of needle.len() to have overlapping matches
            if !P::IS_LONG_PERIOD {
                self.memory = 0; // set to needle.len() - self.period for overlapping matches
            }
            return Some(position..(position + needle.len()));
        }
    }

    #[inline]
    pub(crate) fn next(&mut self, hay: &[T], range: Range<usize>) -> Option<Range<usize>> {
        if self.memory != usize::MAX {
            self.do_next::<ShortPeriod>(hay, range)
        } else {
            self.do_next::<LongPeriod>(hay, range)
        }
    }

    #[inline]
    fn do_next_back<P: Period>(&mut self, hay: &[T], range: Range<usize>) -> Option<Range<usize>> {
        let needle = self.needle;
        let mut end = range.end;
        'search: loop {
            // Check that we have room to search in
            // end - needle.len() will wrap around when there is no more room,
            // but due to slice length limits it can never wrap all the way back
            // into the length of hay.
            if needle.len() + range.start > end {
                return None;
            }
            let front_item = unsafe { hay.get_unchecked(end.wrapping_sub(needle.len())) };

            // Quickly skip by large portions unrelated to our substring
            if !self.byteset_contains(front_item) {
                end -= needle.len();
                if !P::IS_LONG_PERIOD {
                    self.memory_back = needle.len();
                }
                continue 'search;
            }

            // See if the left part of the needle matches
            let crit = if P::IS_LONG_PERIOD {
                self.crit_pos_back
            } else {
                min(self.crit_pos_back, self.memory_back)
            };
            for i in (0..crit).rev() {
                if unsafe { needle.get_unchecked(i) != hay.get_unchecked(end - needle.len() + i) } {
                    end -= self.crit_pos_back - i;
                    if !P::IS_LONG_PERIOD {
                        self.memory_back = needle.len();
                    }
                    continue 'search;
                }
            }

            // See if the right part of the needle matches
            let needle_end = if P::IS_LONG_PERIOD { needle.len() } else { self.memory_back };
            for i in self.crit_pos_back..needle_end {
                if unsafe { needle.get_unchecked(i) != hay.get_unchecked(end - needle.len() + i) } {
                    end -= self.period;
                    if !P::IS_LONG_PERIOD {
                        self.memory_back = self.period;
                    }
                    continue 'search;
                }
            }

            // We have found a match!
            if !P::IS_LONG_PERIOD {
                self.memory_back = needle.len();
            }
            return Some((end - needle.len())..end);
        }
    }

    #[inline]
    pub(crate) fn next_back(&mut self, hay: &[T], range: Range<usize>) -> Option<Range<usize>> {
        if self.memory != usize::MAX {
            self.do_next_back::<ShortPeriod>(hay, range)
        } else {
            self.do_next_back::<LongPeriod>(hay, range)
        }
    }

    #[inline]
    pub fn new(needle: &'p [T]) -> Self {
        let res_lt = T::maximal_suffix(needle, Ordering::Less);
        let res_gt = T::maximal_suffix(needle, Ordering::Greater);
        let (crit_pos, period) = max(res_lt, res_gt);

        let byteset = Self::byteset_create(needle);

        // A particularly readable explanation of what's going on here can be found
        // in Crochemore and Rytter's book "Text Algorithms", ch 13. Specifically
        // see the code for "Algorithm CP" on p. 323.
        //
        // What's going on is we have some critical factorization (u, v) of the
        // needle, and we want to determine whether u is a suffix of
        // &v[..period]. If it is, we use "Algorithm CP1". Otherwise we use
        // "Algorithm CP2", which is optimized for when the period of the needle
        // is large.
        if needle[..crit_pos] == needle[period..(period + crit_pos)] {
            // short period case -- the period is exact
            // compute a separate critical factorization for the reversed needle
            // x = u' v' where |v'| < period(x).
            //
            // This is sped up by the period being known already.
            // Note that a case like x = "acba" may be factored exactly forwards
            // (crit_pos = 1, period = 3) while being factored with approximate
            // period in reverse (crit_pos = 2, period = 2). We use the given
            // reverse factorization but keep the exact period.
            let crit_pos_back = needle.len() - max(
                T::reverse_maximal_suffix(needle, period, Ordering::Greater),
                T::reverse_maximal_suffix(needle, period, Ordering::Less),
            );

            Self {
                crit_pos,
                crit_pos_back,
                period,
                byteset,
                needle,
                memory: 0,
                memory_back: needle.len(),
            }
        } else {
            Self {
                crit_pos,
                crit_pos_back: crit_pos,
                period: max(crit_pos, needle.len() - crit_pos) + 1,
                byteset,
                needle,
                memory: usize::MAX, // Dummy value to signify that the period is long
                memory_back: usize::MAX,
            }
        }
    }

    #[inline]
    fn byteset_create(needle: &[T]) -> FastSkipByteset {
        needle.iter().fold(0, |a, b| b.byteset_mask() | a)
    }
    #[inline]
    fn byteset_contains(&self, item: &T) -> bool {
        (self.byteset & item.byteset_mask()) != 0
    }
}

unsafe impl<'p, T> Searcher<[T]> for TwoWaySearcher<'p, T>
where
    T: PartialEq + 'p,
{
    #[inline]
    fn search(&mut self, span: Span<&[T]>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        self.next(hay, range)
    }
}

unsafe impl<'p, T> ReverseSearcher<[T]> for TwoWaySearcher<'p, T>
where
    T: PartialEq + 'p,
{
    #[inline]
    fn rsearch(&mut self, span: Span<&[T]>) -> Option<Range<usize>> {
        let (hay, range) = span.into_parts();
        self.next_back(hay, range)
    }
}

//------------------------------------------------------------------------------
// Naive (state-less) searcher
//------------------------------------------------------------------------------

#[derive(Debug)]
pub struct NaiveSearcher<'p, T: 'p>(&'p [T]);

impl<'p, T: 'p> Clone for NaiveSearcher<'p, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'p, T: 'p> Copy for NaiveSearcher<'p, T> {}

unsafe impl<'p, T> Consumer<[T]> for NaiveSearcher<'p, T>
where
    T: PartialEq + 'p,
{
    #[inline]
    fn consume(&mut self, span: Span<&[T]>) -> Option<usize> {
        let (hay, range) = span.into_parts();
        let check_end = range.start + self.0.len();
        if range.end < check_end {
            return None;
        }
        if unsafe { hay.get_unchecked(range.start..check_end) } == self.0 {
            Some(check_end)
        } else {
            None
        }
    }
}

unsafe impl<'p, T> ReverseConsumer<[T]> for NaiveSearcher<'p, T>
where
    T: PartialEq + 'p,
{
    #[inline]
    fn rconsume(&mut self, span: Span<&[T]>) -> Option<usize> {
        let (hay, range) = span.into_parts();
        if range.start + self.0.len() > range.end {
            return None;
        }
        let index = range.end - self.0.len();
        if unsafe { hay.get_unchecked(index..range.end) } == self.0 {
            Some(index)
        } else {
            None
        }
    }
}

impl<'p, T: 'p> NaiveSearcher<'p, T> {
    #[inline]
    pub fn new(slice: &'p [T]) -> Self {
        NaiveSearcher(slice)
    }

    #[inline]
    pub fn needle(&self) -> &'p [T] {
        self.0
    }
}

//------------------------------------------------------------------------------
// Slice searcher
//------------------------------------------------------------------------------

#[derive(Debug)]
pub enum SliceSearcher<'p, T: 'p> {
    TwoWay(TwoWaySearcher<'p, T>),
    Empty(EmptySearcher),
}

impl<'p, T: PartialEq + 'p> SliceSearcher<'p, T> {
    #[inline]
    pub fn new(slice: &'p [T]) -> Self {
        if slice.is_empty() {
            SliceSearcher::Empty(EmptySearcher::default())
        } else {
            SliceSearcher::TwoWay(TwoWaySearcher::new(slice))
        }
    }

    #[inline]
    pub fn needle(&self) -> &'p [T] {
        match self {
            SliceSearcher::TwoWay(s) => s.needle,
            SliceSearcher::Empty(_) => &[],
        }
    }
}

impl<'p, T: 'p> Clone for SliceSearcher<'p, T> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            SliceSearcher::TwoWay(s) => SliceSearcher::TwoWay(*s),
            SliceSearcher::Empty(s) => SliceSearcher::Empty(s.clone()),
        }
    }
}

macro_rules! forward {
    (searcher: $self:expr, $s:ident => $e:expr) => {
        match $self {
            SliceSearcher::TwoWay($s) => $e,
            SliceSearcher::Empty($s) => $e,
        }
    };
}

unsafe impl<'p, T, A> Searcher<A> for SliceSearcher<'p, T>
where
    A: Hay<Index = usize> + ?Sized,
    TwoWaySearcher<'p, T>: Searcher<A>,
{
    #[inline]
    fn search(&mut self, span: Span<&A>) -> Option<Range<usize>> {
        forward!(searcher: self, s => s.search(span))
    }
}

unsafe impl<'p, T, A> ReverseSearcher<A> for SliceSearcher<'p, T>
where
    A: Hay<Index = usize> + ?Sized,
    TwoWaySearcher<'p, T>: ReverseSearcher<A>,
{
    #[inline]
    fn rsearch(&mut self, span: Span<&A>) -> Option<Range<usize>> {
        forward!(searcher: self, s => s.rsearch(span))
    }
}

macro_rules! impl_needle_for_slice_searcher {
    ([$($gen:tt)+] <$haystack:ty> for $ty:ty) => {
        impl<$($gen)+, 'h, T> Needle<$haystack> for $ty
        where
            T: PartialEq + 'p,
        {
            type Searcher = SliceSearcher<'p, T>;
            type Consumer = NaiveSearcher<'p, T>;

            #[inline]
            fn into_searcher(self) -> Self::Searcher {
                SliceSearcher::new(self)
            }

            #[inline]
            fn into_consumer(self) -> Self::Consumer {
                NaiveSearcher::new(self)
            }
        }
    };

    ($($index:expr),*) => {
        impl_needle_for_slice_searcher!(['p] <&'h [T]> for &'p [T]);
        impl_needle_for_slice_searcher!(['p] <&'h mut [T]> for &'p [T]);
        impl_needle_for_slice_searcher!(['q, 'p] <&'h [T]> for &'q &'p [T]);
        impl_needle_for_slice_searcher!(['q, 'p] <&'h mut [T]> for &'q &'p [T]);
        $(
            impl_needle_for_slice_searcher!(['p] <&'h [T]> for &'p [T; $index]);
            impl_needle_for_slice_searcher!(['p] <&'h mut [T]> for &'p [T; $index]);
        )*
    }
}

impl_needle_for_slice_searcher!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
);
