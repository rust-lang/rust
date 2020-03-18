use {
    crate::TextSize,
    std::{
        cmp, fmt,
        ops::{Bound, Index, IndexMut, Range, RangeBounds, RangeFrom},
    },
};

/// A range in text, represented as a pair of [`TextSize`][struct@TextSize].
///
/// # Translation from `text_unit`
///
/// - `TextRange::from_to(from, to)`        ⟹ `TextRange(from, to)`
/// - `TextRange::offset_len(offset, size)` ⟹ `TextRange::before(size).offset(offset)`
/// - `range.start()`                       ⟹ `range.start()`
/// - `range.end()`                         ⟹ `range.end()`
/// - `range.len()`                         ⟹ `range.len()`
/// - `range.is_empty()`                    ⟹ `range.is_empty()`
/// - `a.is_subrange(b)`                    ⟹ `b.contains_range(a)`
/// - `a.intersection(b)`                   ⟹ `TextRange::intersection(a, b)`
/// - `a.extend_to(b)`                      ⟹ `TextRange::covering(a, b)`
/// - `range.contains(offset)`              ⟹ `range.contains(point)`
/// - `range.contains_inclusive(offset)`    ⟹ `range.contains_inclusive(point)`
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct TextRange {
    // Invariant: start <= end
    start: TextSize,
    end: TextSize,
}

impl fmt::Debug for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start().raw, self.end().raw)
    }
}

/// Creates a new `TextRange` with the given `start` and `end` (`start..end`).
///
/// # Panics
///
/// Panics if `end < start`.
#[allow(non_snake_case)]
#[inline]
pub fn TextRange(start: TextSize, end: TextSize) -> TextRange {
    assert!(start <= end);
    TextRange { start, end }
}

impl TextRange {
    /// Create a zero-length range at the specified offset (`offset..offset`).
    #[inline]
    pub const fn empty(offset: TextSize) -> TextRange {
        TextRange {
            start: offset,
            end: offset,
        }
    }

    /// Create a range up to the given end (`..end`).
    #[inline]
    pub const fn before(end: TextSize) -> TextRange {
        TextRange {
            start: TextSize::zero(),
            end,
        }
    }

    /// Offset this range by some amount.
    ///
    /// This is typically used to convert a range from one coordinate space to
    /// another, such as from within a substring to within an entire document.
    #[inline]
    pub fn offset(self, offset: TextSize) -> TextRange {
        TextRange(
            self.start().checked_add(offset).unwrap(),
            self.end().checked_add(offset).unwrap(),
        )
    }
}

/// Identity methods.
impl TextRange {
    /// The start point of this range.
    #[inline]
    pub const fn start(self) -> TextSize {
        self.start
    }

    /// The end point of this range.
    #[inline]
    pub const fn end(self) -> TextSize {
        self.end
    }

    /// The size of this range.
    #[inline]
    pub const fn len(self) -> TextSize {
        // HACK for const fn: math on primitives only
        TextSize(self.end().raw - self.start().raw)
    }

    /// Check if this range is empty.
    #[inline]
    pub const fn is_empty(self) -> bool {
        // HACK for const fn: math on primitives only
        self.start().raw == self.end().raw
    }
}

/// Manipulation methods.
impl TextRange {
    /// Check if this range contains an offset.
    ///
    /// The end index is considered excluded.
    pub fn contains(self, offset: TextSize) -> bool {
        self.start() <= offset && offset < self.end()
    }

    /// Check if this range contains an offset.
    ///
    /// The end index is considered included.
    pub fn contains_inclusive(self, offset: TextSize) -> bool {
        self.start() <= offset && offset <= self.end()
    }

    /// Check if this range completely contains another range.
    pub fn contains_range(self, other: TextRange) -> bool {
        self.start() <= other.start() && other.end() <= self.end()
    }

    /// The range covered by both ranges, if it exists.
    /// If the ranges touch but do not overlap, the output range is empty.
    pub fn intersection(lhs: TextRange, rhs: TextRange) -> Option<TextRange> {
        let start = cmp::max(lhs.start(), rhs.start());
        let end = cmp::min(lhs.end(), rhs.end());
        if end < start {
            return None;
        }
        Some(TextRange(start, end))
    }

    /// The smallest range that completely contains both ranges.
    pub fn covering(lhs: TextRange, rhs: TextRange) -> TextRange {
        let start = cmp::min(lhs.start(), rhs.start());
        let end = cmp::max(lhs.end(), rhs.end());
        TextRange(start, end)
    }
}

impl Index<TextRange> for str {
    type Output = str;
    #[inline]
    fn index(&self, index: TextRange) -> &Self::Output {
        &self[Range::<usize>::from(index)]
    }
}

impl IndexMut<TextRange> for str {
    #[inline]
    fn index_mut(&mut self, index: TextRange) -> &mut Self::Output {
        &mut self[Range::<usize>::from(index)]
    }
}

impl RangeBounds<TextSize> for TextRange {
    fn start_bound(&self) -> Bound<&TextSize> {
        Bound::Included(&self.start)
    }

    fn end_bound(&self) -> Bound<&TextSize> {
        Bound::Excluded(&self.end)
    }
}

impl<T> From<TextRange> for Range<T>
where
    T: From<TextSize>,
{
    #[inline]
    fn from(r: TextRange) -> Self {
        r.start().into()..r.end().into()
    }
}
