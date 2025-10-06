use std::fmt::Debug;
use std::hash::Hash;
use std::ops;
use std::slice::SliceIndex;

/// Represents some newtyped `usize` wrapper.
///
/// Purpose: avoid mixing indexes for different bitvector domains.
pub trait Idx: Copy + 'static + Eq + PartialEq + Debug + Hash {
    fn new(idx: usize) -> Self;

    fn index(self) -> usize;

    #[inline]
    fn increment_by(&mut self, amount: usize) {
        *self = self.plus(amount);
    }

    #[inline]
    #[must_use = "Use `increment_by` if you wanted to update the index in-place"]
    fn plus(self, amount: usize) -> Self {
        Self::new(self.index() + amount)
    }
}

impl Idx for usize {
    #[inline]
    fn new(idx: usize) -> Self {
        idx
    }
    #[inline]
    fn index(self) -> usize {
        self
    }
}

impl Idx for u32 {
    #[inline]
    fn new(idx: usize) -> Self {
        assert!(idx <= u32::MAX as usize);
        idx as u32
    }
    #[inline]
    fn index(self) -> usize {
        self as usize
    }
}

/// Helper trait for indexing operations with a custom index type.
pub trait IntoSliceIdx<I, T: ?Sized> {
    type Output: SliceIndex<T>;
    fn into_slice_idx(self) -> Self::Output;
}

impl<I: Idx, T> IntoSliceIdx<I, [T]> for I {
    type Output = usize;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        self.index()
    }
}

impl<I, T> IntoSliceIdx<I, [T]> for ops::RangeFull {
    type Output = ops::RangeFull;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        self
    }
}

impl<I: Idx, T> IntoSliceIdx<I, [T]> for ops::Range<I> {
    type Output = ops::Range<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        ops::Range { start: self.start.index(), end: self.end.index() }
    }
}

impl<I: Idx, T> IntoSliceIdx<I, [T]> for ops::RangeFrom<I> {
    type Output = ops::RangeFrom<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        ops::RangeFrom { start: self.start.index() }
    }
}

impl<I: Idx, T> IntoSliceIdx<I, [T]> for ops::RangeTo<I> {
    type Output = ops::RangeTo<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        ..self.end.index()
    }
}

impl<I: Idx, T> IntoSliceIdx<I, [T]> for ops::RangeInclusive<I> {
    type Output = ops::RangeInclusive<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        ops::RangeInclusive::new(self.start().index(), self.end().index())
    }
}

impl<I: Idx, T> IntoSliceIdx<I, [T]> for ops::RangeToInclusive<I> {
    type Output = ops::RangeToInclusive<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        ..=self.end.index()
    }
}

#[cfg(feature = "nightly")]
impl<I: Idx, T> IntoSliceIdx<I, [T]> for core::range::Range<I> {
    type Output = core::range::Range<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        core::range::Range { start: self.start.index(), end: self.end.index() }
    }
}

#[cfg(feature = "nightly")]
impl<I: Idx, T> IntoSliceIdx<I, [T]> for core::range::RangeFrom<I> {
    type Output = core::range::RangeFrom<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        core::range::RangeFrom { start: self.start.index() }
    }
}

#[cfg(feature = "nightly")]
impl<I: Idx, T> IntoSliceIdx<I, [T]> for core::range::RangeInclusive<I> {
    type Output = core::range::RangeInclusive<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        core::range::RangeInclusive { start: self.start.index(), last: self.last.index() }
    }
}

#[cfg(all(feature = "nightly", not(bootstrap)))]
impl<I: Idx, T> IntoSliceIdx<I, [T]> for core::range::RangeToInclusive<I> {
    type Output = core::range::RangeToInclusive<usize>;
    #[inline]
    fn into_slice_idx(self) -> Self::Output {
        core::range::RangeToInclusive { last: self.last.index() }
    }
}
