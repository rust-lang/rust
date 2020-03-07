use {
    crate::{TextSize, TextSized},
    std::{
        cmp,
        convert::{TryFrom, TryInto},
        fmt,
        num::TryFromIntError,
        ops::{
            Add, AddAssign, Bound, Index, IndexMut, Range, RangeBounds, RangeInclusive, RangeTo,
            RangeToInclusive, Sub, SubAssign,
        },
    },
};

/// A range in text, represented as a pair of [`TextSize`][struct@TextSize].
///
/// It is a logical error to have `end() < start()`, but
/// code must not assume this is true for `unsafe` guarantees.
///
/// # Translation from `text_unit`
///
/// - `TextRange::from_to(from, to)` ⟹ `TextRange::from(from..to)`
/// - `TextRange::offset_len(offset, size)` ⟹ `TextRange::at(offset).with_len(size)`
/// - `range.start()` ⟹ `range.start()`
/// - `range.end()` ⟹ `range.end()`
/// - `range.len()` ⟹ `range.len()`<sup>†</sup>
/// - `range.is_empty()` ⟹ `range.is_empty()`
/// - `a.is_subrange(b)` ⟹ `b.contains(a)`
/// - `a.intersection(b)` ⟹ `TextRange::intersection(a, b)`
/// - `a.extend_to(b)` ⟹ `TextRange::covering(a, b)`
/// - `range.contains(offset)` ⟹ `range.contains_point(point)`
/// - `range.contains_inclusive(offset)` ⟹ `range.contains_point_inclusive(point)`
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
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TextRange {
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
    /// A range covering the text size of some text-like object.
    pub fn of(size: impl TextSized) -> TextRange {
        TextRange(0.into(), size.text_size())
    }

    /// An empty range at some text size offset.
    pub fn at(size: impl Into<TextSize>) -> TextRange {
        let size = size.into();
        TextRange(size, size)
    }

    /// Set the length without changing the starting offset.
    pub fn with_len(self, len: impl Into<TextSize>) -> TextRange {
        TextRange(self.start(), self.start() + len.into())
    }

    /// Set the starting offset without changing the length.
    pub fn with_offset(self, offset: impl Into<TextSize>) -> TextRange {
        TextRange::at(offset).with_len(self.len())
    }

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
    pub fn contains_point(self, point: impl Into<TextSize>) -> bool {
        let point = point.into();
        self.start() <= point && point < self.end()
    }

    /// Check if this range contains a point.
    ///
    /// The end index is considered included.
    pub fn contains_point_inclusive(self, point: impl Into<TextSize>) -> bool {
        let point = point.into();
        self.start() <= point && point <= self.end()
    }

    /// Offset the entire range by some text size.
    pub fn checked_add(self, rhs: impl TryInto<TextSize>) -> Option<TextRange> {
        let rhs = rhs.try_into().ok()?;
        Some(TextRange(
            self.start().checked_add(rhs)?,
            self.end().checked_add(rhs)?,
        ))
    }

    /// Offset the entire range by some text size.
    pub fn checked_sub(self, rhs: impl TryInto<TextSize>) -> Option<TextRange> {
        let rhs = rhs.try_into().ok()?;
        Some(TextRange(
            self.start().checked_sub(rhs)?,
            self.end().checked_sub(rhs)?,
        ))
    }
}

impl Index<TextRange> for str {
    type Output = str;
    fn index(&self, index: TextRange) -> &Self::Output {
        &self[index.start().ix()..index.end().ix()]
    }
}

impl IndexMut<TextRange> for str {
    fn index_mut(&mut self, index: TextRange) -> &mut Self::Output {
        &mut self[index.start().ix()..index.end().ix()]
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
        impl TryFrom<RangeInclusive<$lte>> for TextRange {
            type Error = TryFromIntError;
            fn try_from(value: RangeInclusive<$lte>) -> Result<TextRange, Self::Error> {
                let (start, end) = value.into_inner();
                let end: TextSize = end.into();
                // This is the only way to get a TryFromIntError currently.
                let end = end.checked_add(1).ok_or_else(|| u8::try_from(-1).unwrap_err())?;
                Ok(TextRange(start.into(), end))
            }
        }
        impl From<RangeTo<$lte>> for TextRange {
            fn from(value: RangeTo<$lte>) -> TextRange {
                TextRange(0.into(), value.end.into())
            }
        }
        impl TryFrom<RangeToInclusive<$lte>> for TextRange {
            type Error = TryFromIntError;
            fn try_from(value: RangeToInclusive<$lte>) -> Result<TextRange, Self::Error> {
                let start: TextSize = 0.into();
                let end: TextSize = value.end.into();
                TextRange::try_from(start..=end)
            }
        }
    };
    (TryFrom<$gt:ident> for TextRange) => {
        impl TryFrom<Range<$gt>> for TextRange {
            type Error = <$gt as TryInto<u32>>::Error;
            fn try_from(value: Range<$gt>) -> Result<TextRange, Self::Error> {
                Ok(TextRange(value.start.try_into()?, value.end.try_into()?))
            }
        }
        impl TryFrom<RangeInclusive<$gt>> for TextRange {
            type Error = TryFromIntError;
            fn try_from(value: RangeInclusive<$gt>) -> Result<TextRange, Self::Error> {
                let (start, end) = value.into_inner();
                let end: TextSize = end.try_into()?;
                // This is the only way to get a TryFromIntError currently.
                let end = end.checked_add(1).ok_or_else(|| u8::try_from(-1).unwrap_err())?;
                Ok(TextRange(start.try_into()?, end))
            }
        }
        impl TryFrom<RangeTo<$gt>> for TextRange {
            type Error = TryFromIntError;
            fn try_from(value: RangeTo<$gt>) -> Result<TextRange, Self::Error> {
                Ok(TextRange(0.into(), value.end.try_into()?))
            }
        }
        impl TryFrom<RangeToInclusive<$gt>> for TextRange {
            type Error = TryFromIntError;
            fn try_from(value: RangeToInclusive<$gt>) -> Result<TextRange, Self::Error> {
                let start: TextSize = 0.into();
                let end: TextSize = value.end.try_into()?;
                TextRange::try_from(start..=end)
            }
        }
    };
    {
        lt TextSize [$($lt:ident)*]
        eq TextSize [$($eq:ident)*]
        gt TextSize [$($gt:ident)*]
        varries     [$($var:ident)*]
    } => {
        $(
            // Not `From` yet because of integer type fallback. We want e.g.
            // `TextRange::from(0)` and `range + 1` to work, and more `From`
            // impls means that this will try (and fail) to use i32 rather
            // than one of the unsigned integer types that actually work.
            conversions!(TryFrom<$lt> for TextRange);
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

impl Into<TextRange> for &'_ TextRange {
    fn into(self) -> TextRange {
        *self
    }
}

impl Into<TextRange> for &'_ mut TextRange {
    fn into(self) -> TextRange {
        *self
    }
}

macro_rules! op {
    (impl $Op:ident for TextRange by fn $f:ident = $op:tt) => {
        impl<IntoSize: Copy> $Op<IntoSize> for TextRange
        where
            TextSize: $Op<IntoSize, Output = TextSize>,
        {
            type Output = TextRange;
            fn $f(self, rhs: IntoSize) -> TextRange {
                TextRange(self.start() $op rhs, self.end() $op rhs)
            }
        }
        impl<IntoSize> $Op<IntoSize> for &'_ TextRange
        where
            TextRange: $Op<IntoSize, Output = TextRange>,
        {
            type Output = TextRange;
            fn $f(self, rhs: IntoSize) -> TextRange {
                *self $op rhs
            }
        }
        impl<IntoSize> $Op<IntoSize> for &'_ mut TextRange
        where
            TextRange: $Op<IntoSize, Output = TextRange>,
        {
            type Output = TextRange;
            fn $f(self, rhs: IntoSize) -> TextRange {
                *self $op rhs
            }
        }
    };
}

op!(impl Add for TextRange by fn add = +);
op!(impl Sub for TextRange by fn sub = -);

impl<A> AddAssign<A> for TextRange
where
    TextRange: Add<A, Output = TextRange>,
{
    fn add_assign(&mut self, rhs: A) {
        *self = *self + rhs
    }
}

impl<S> SubAssign<S> for TextRange
where
    TextRange: Sub<S, Output = TextRange>,
{
    fn sub_assign(&mut self, rhs: S) {
        *self = *self - rhs
    }
}
