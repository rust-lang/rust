use {
    crate::TextSize,
    std::{
        cmp,
        convert::{TryFrom, TryInto},
        fmt,
        ops::{Bound, Index, IndexMut, Range, RangeBounds},
    },
};

/// A range in text, represented as a pair of [`TextSize`][struct@TextSize].
///
/// It is a logical error to have `end() < start()`, but
/// code must not assume this is true for `unsafe` guarantees.
///
/// # Translation from `text_unit`
///
/// - `TextRange::from_to(from, to)`        ⟹ `TextRange::from(from..to)`
/// - `TextRange::offset_len(offset, size)` ⟹ `TextRange::from(offset..offset + size)`
/// - `range.start()`                       ⟹ `range.start()`
/// - `range.end()`                         ⟹ `range.end()`
/// - `range.len()`                         ⟹ `range.len()`<sup>†</sup>
/// - `range.is_empty()`                    ⟹ `range.is_empty()`
/// - `a.is_subrange(b)`                    ⟹ `b.contains(a)`
/// - `a.intersection(b)`                   ⟹ `TextRange::intersection(a, b)`
/// - `a.extend_to(b)`                      ⟹ `TextRange::covering(a, b)`
/// - `range.contains(offset)`              ⟹ `range.contains_exclusive(point)`
/// - `range.contains_inclusive(offset)`    ⟹ `range.contains_inclusive(point)`
///
/// † See the note on [`TextRange::len`] for differing behavior for incorrect reverse ranges.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct TextRange {
    start: TextSize,
    end: TextSize,
}

#[allow(non_snake_case)]
pub(crate) const fn TextRange(start: TextSize, end: TextSize) -> TextRange {
    TextRange { start, end }
}

impl fmt::Debug for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{})", self.start(), self.end())
    }
}

/// Identity methods.
impl TextRange {
    /// The start point of this range.
    pub const fn start(self) -> TextSize {
        self.start
    }

    /// The end point of this range.
    pub const fn end(self) -> TextSize {
        self.end
    }

    /// The size of this range.
    ///
    /// # Panics
    ///
    /// When `end() < start()`, triggers a subtraction overflow.
    /// This will panic with debug assertions, and overflow without.
    pub const fn len(self) -> TextSize {
        // HACK for const fn: math on primitives only
        TextSize(self.end().raw - self.start().raw)
    }

    /// Check if this range empty or reversed.
    ///
    /// When `end() < start()`, this returns false.
    /// Code should prefer `is_empty()` to `len() == 0`,
    /// as this safeguards against incorrect reverse ranges.
    pub const fn is_empty(self) -> bool {
        // HACK for const fn: math on primitives only
        self.start().raw >= self.end().raw
    }
}

/// Manipulation methods.
impl TextRange {
    /// Check if this range completely contains another range.
    pub fn contains(self, other: TextRange) -> bool {
        self.start() <= other.start() && other.end() <= self.end()
    }

    /// The range covered by both ranges, if it exists.
    /// If the ranges touch but do not overlap, the output range is empty.
    pub fn intersection(lhs: TextRange, rhs: TextRange) -> Option<TextRange> {
        let start = cmp::max(lhs.start(), rhs.start());
        let end = cmp::min(lhs.end(), rhs.end());
        Some(TextRange(start, end)).filter(|_| start <= end)
    }

    /// The smallest range that completely contains both ranges.
    pub fn covering(lhs: TextRange, rhs: TextRange) -> TextRange {
        let start = cmp::min(lhs.start(), rhs.start());
        let end = cmp::max(lhs.end(), rhs.end());
        TextRange(start, end)
    }

    /// Check if this range contains a point.
    ///
    /// The end index is considered excluded.
    pub fn contains_exclusive(self, point: impl Into<TextSize>) -> bool {
        let point = point.into();
        self.start() <= point && point < self.end()
    }

    /// Check if this range contains a point.
    ///
    /// The end index is considered included.
    pub fn contains_inclusive(self, point: impl Into<TextSize>) -> bool {
        let point = point.into();
        self.start() <= point && point <= self.end()
    }
}

fn ix(size: TextSize) -> usize {
    size.try_into()
        .unwrap_or_else(|_| panic!("overflow when converting TextSize to usize index"))
}

impl Index<TextRange> for str {
    type Output = str;
    fn index(&self, index: TextRange) -> &Self::Output {
        &self[ix(index.start())..ix(index.end())]
    }
}

impl IndexMut<TextRange> for str {
    fn index_mut(&mut self, index: TextRange) -> &mut Self::Output {
        &mut self[ix(index.start())..ix(index.end())]
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

macro_rules! conversions {
    (From<$lte:ident> for TextRange) => {
        impl From<Range<$lte>> for TextRange {
            fn from(value: Range<$lte>) -> TextRange {
                TextRange(value.start.into(), value.end.into())
            }
        }
        // Just support `start..end` for now, not `..end`, `start..=end`, `..=end`.
    };
    (TryFrom<$gt:ident> for TextRange) => {
        impl TryFrom<Range<$gt>> for TextRange {
            type Error = <$gt as TryInto<u32>>::Error;
            fn try_from(value: Range<$gt>) -> Result<TextRange, Self::Error> {
                Ok(TextRange(value.start.try_into()?, value.end.try_into()?))
            }
        }
        // Just support `start..end` for now, not `..end`, `start..=end`, `..=end`.
    };
    {
        lt TextSize [$($lt:ident)*]
        eq TextSize [$($eq:ident)*]
        gt TextSize [$($gt:ident)*]
        varries     [$($var:ident)*]
    } => {
        $(
            conversions!(From<$lt> for TextRange);
            // unlike TextSize, we do not provide conversions in the "out" direction.
        )*

        $(
            conversions!(From<$eq> for TextRange);
        )*

        $(
            conversions!(TryFrom<$gt> for TextRange);
        )*

        $(
            conversions!(TryFrom<$var> for TextRange);
        )*
    };
}

// FIXME: when `default impl` is usable, change to blanket impls for [Try]Into<TextSize> instead
conversions! {
    lt TextSize [u8 u16]
    eq TextSize [u32 TextSize]
    gt TextSize [u64]
    varries     [usize]
}
