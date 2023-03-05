use crate::intrinsics::{assert_unsafe_precondition, unchecked_add, unchecked_sub};
use crate::iter::{FusedIterator, TrustedLen};
use crate::ops::Range;

/// Represents a *canonical* range of indexes, with a safety invariant that `start <= end`.
///
/// This allows some μoptimizations:
/// - Slice indexing can check just that `end` is in-bounds, and
///   trust that means that `start` will also be in-bounds.
/// - [`len()`](Self::len) can trust that `end - start` cannot overflow.
///
/// (Normal `Range` code needs to handle degenerate ranges like `10..0`,
///  which takes extra checks compared to only handling the canonical form.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IndexRange {
    start: usize,
    end: usize,
}

impl IndexRange {
    #[inline(always)]
    pub(crate) const fn from_range(Range { start, end }: Range<usize>) -> Option<Self> {
        if start <= end { Some(IndexRange { start, end }) } else { None }
    }

    /// # Safety
    /// The caller must ensure that `start <= end`.
    #[inline(always)]
    pub(crate) const unsafe fn from_range_unchecked(Range { start, end }: Range<usize>) -> Self {
        // SAFETY: Same precondition
        unsafe { Self::new_unchecked(start, end) }
    }

    /// Creates an `IndexRange` from its `start` and `end` indices,
    /// without checking whether they're ordered correctly.
    ///
    /// Unlike `Range<usize>`, an `IndexRange` does not allow non-canonical empty
    /// range (like `10..2`).  The only allowed empty ranges are when
    /// `start == end`, such as `0..0` or `10..10`.
    ///
    /// # Safety
    /// The caller must ensure that `start <= end`.
    #[inline]
    pub const unsafe fn new_unchecked(start: usize, end: usize) -> Self {
        // SAFETY: comparisons on usize are pure
        unsafe {
            assert_unsafe_precondition!(
               "IndexRange::new_unchecked requires `start <= end`",
                (start: usize, end: usize) => start <= end
            )
        };
        IndexRange { start, end }
    }

    #[inline]
    pub const fn zero_to(end: usize) -> Self {
        IndexRange { start: 0, end }
    }

    #[inline]
    pub const fn start(&self) -> usize {
        self.start
    }

    #[inline]
    pub const fn end(&self) -> usize {
        self.end
    }

    #[inline]
    pub const fn len(&self) -> usize {
        // SAFETY: By invariant, this cannot wrap
        unsafe { unchecked_sub(self.end, self.start) }
    }

    /// # Safety
    /// - Can only be called when `start < end`, aka when `len > 0`.
    #[inline]
    unsafe fn next_unchecked(&mut self) -> usize {
        debug_assert!(self.start < self.end);

        let value = self.start;
        // SAFETY: The range isn't empty, so this cannot overflow
        self.start = unsafe { unchecked_add(value, 1) };
        value
    }

    /// # Safety
    /// - Can only be called when `start < end`, aka when `len > 0`.
    #[inline]
    unsafe fn next_back_unchecked(&mut self) -> usize {
        debug_assert!(self.start < self.end);

        // SAFETY: The range isn't empty, so this cannot overflow
        let value = unsafe { unchecked_sub(self.end, 1) };
        self.end = value;
        value
    }

    /// Removes the first `n` items from this range, returning them as an `IndexRange`.
    /// If there are fewer than `n`, then the whole range is returned and
    /// `self` is left empty.
    ///
    /// This is designed to help implement `Iterator::advance_by`.
    #[inline]
    pub fn take_prefix(&mut self, n: usize) -> Self {
        let mid = if n <= self.len() {
            // SAFETY: We just checked that this will be between start and end,
            // and thus the addition cannot overflow.
            unsafe { unchecked_add(self.start, n) }
        } else {
            self.end
        };
        let prefix = Self { start: self.start, end: mid };
        self.start = mid;
        prefix
    }

    /// Removes the last `n` items from this range, returning them as an `IndexRange`.
    /// If there are fewer than `n`, then the whole range is returned and
    /// `self` is left empty.
    ///
    /// This is designed to help implement `Iterator::advance_back_by`.
    #[inline]
    pub fn take_suffix(&mut self, n: usize) -> Self {
        let mid = if n <= self.len() {
            // SAFETY: We just checked that this will be between start and end,
            // and thus the addition cannot overflow.
            unsafe { unchecked_sub(self.end, n) }
        } else {
            self.start
        };
        let suffix = Self { start: mid, end: self.end };
        self.end = mid;
        suffix
    }
}

impl Iterator for IndexRange {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.len() > 0 {
            // SAFETY: We just checked that the range is non-empty
            unsafe { Some(self.next_unchecked()) }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn advance_by(&mut self, n: usize) -> Result<(), usize> {
        let original_len = self.len();
        self.take_prefix(n);
        if n > original_len { Err(original_len) } else { Ok(()) }
    }
}

impl DoubleEndedIterator for IndexRange {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        if self.len() > 0 {
            // SAFETY: We just checked that the range is non-empty
            unsafe { Some(self.next_back_unchecked()) }
        } else {
            None
        }
    }

    #[inline]
    fn advance_back_by(&mut self, n: usize) -> Result<(), usize> {
        let original_len = self.len();
        self.take_suffix(n);
        if n > original_len { Err(original_len) } else { Ok(()) }
    }
}

impl ExactSizeIterator for IndexRange {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
}

// SAFETY: Because we only deal in `usize`, our `len` is always perfect.
unsafe impl TrustedLen for IndexRange {}

impl FusedIterator for IndexRange {}
