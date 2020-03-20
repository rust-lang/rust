use {
    crate::TextSize,
    std::{
        cmp, fmt,
        ops::{Add, AddAssign, Bound, Index, IndexMut, Range, RangeBounds, Sub, SubAssign},
    },
};

/// A range in text, represented as a pair of [`TextSize`][struct@TextSize].
///
/// # Translation from `text_unit`
///
/// - `TextRange::from_to(from, to)`        ⟹ `TextRange(from, to)`
/// - `TextRange::offset_len(offset, size)` ⟹ `TextRange::up_to(size) + offset`
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
    pub const fn up_to(end: TextSize) -> TextRange {
        TextRange {
            start: TextSize::zero(),
            end,
        }
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
    pub fn intersect(self, other: TextRange) -> Option<TextRange> {
        let start = cmp::max(self.start(), other.start());
        let end = cmp::min(self.end(), other.end());
        if end < start {
            return None;
        }
        Some(TextRange(start, end))
    }

    /// Extends the range to cover `other` as well.
    pub fn cover(self, other: TextRange) -> TextRange {
        let start = cmp::min(self.start(), other.start());
        let end = cmp::max(self.end(), other.end());
        TextRange(start, end)
    }

    /// Extends the range to cover `other` offsets as well.
    pub fn cover_offset(self, offset: TextSize) -> TextRange {
        self.cover(TextRange::empty(offset))
    }

    /// Add an offset to this range.
    ///
    /// Note that this is not appropriate for changing where a `TextRange` is
    /// within some string; rather, it is for changing the reference anchor
    /// that the `TextRange` is measured against.
    ///
    /// The unchecked version (`Add::add`) will _always_ panic on overflow,
    /// in contrast to primitive integers, which check in debug mode only.
    #[inline]
    pub fn checked_add(self, offset: TextSize) -> Option<TextRange> {
        Some(TextRange {
            start: self.start.checked_add(offset)?,
            end: self.end.checked_add(offset)?,
        })
    }

    /// Subtract an offset from this range.
    ///
    /// Note that this is not appropriate for changing where a `TextRange` is
    /// within some string; rather, it is for changing the reference anchor
    /// that the `TextRange` is measured against.
    ///
    /// The unchecked version (`Sub::sub`) will _always_ panic on overflow,
    /// in contrast to primitive integers, which check in debug mode only.
    #[inline]
    pub fn checked_sub(self, offset: TextSize) -> Option<TextRange> {
        Some(TextRange {
            start: self.start.checked_sub(offset)?,
            end: self.end.checked_sub(offset)?,
        })
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

macro_rules! ops {
    (impl $Op:ident for TextRange by fn $f:ident = $op:tt) => {
        impl $Op<&TextSize> for TextRange {
            type Output = TextRange;
            #[inline]
            fn $f(self, other: &TextSize) -> TextRange {
                self $op *other
            }
        }
        impl<T> $Op<T> for &TextRange
        where
            TextRange: $Op<T, Output=TextRange>,
        {
            type Output = TextRange;
            #[inline]
            fn $f(self, other: T) -> TextRange {
                *self $op other
            }
        }
    };
}

impl Add<TextSize> for TextRange {
    type Output = TextRange;
    #[inline]
    fn add(self, offset: TextSize) -> TextRange {
        self.checked_add(offset)
            .expect("TextRange +offset overflowed")
    }
}

impl Sub<TextSize> for TextRange {
    type Output = TextRange;
    #[inline]
    fn sub(self, offset: TextSize) -> TextRange {
        self.checked_sub(offset)
            .expect("TextRange -offset overflowed")
    }
}

ops!(impl Add for TextRange by fn add = +);
ops!(impl Sub for TextRange by fn sub = -);

impl<A> AddAssign<A> for TextRange
where
    TextRange: Add<A, Output = TextRange>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        *self = *self + rhs
    }
}

impl<S> SubAssign<S> for TextRange
where
    TextRange: Sub<S, Output = TextRange>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: S) {
        *self = *self - rhs
    }
}
