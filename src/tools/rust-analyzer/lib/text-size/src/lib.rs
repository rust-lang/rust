#[cfg(feature = "serde")]
extern crate serde;

use std::{fmt, iter, ops};

/// An offset into text.
/// Offset is represented as `u32` storing number of utf8-bytes,
/// but most of the clients should treat it like opaque measure.
// BREAK:  TextSize(u32)
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TextUnit(u32);

impl TextUnit {
    // BREAK: consider renaming?
    /// `TextUnit` equal to the length of this char.
    #[inline(always)]
    pub fn of_char(c: char) -> TextUnit {
        TextUnit(c.len_utf8() as u32)
    }

    // BREAK: consider renaming?
    /// `TextUnit` equal to the length of this string.
    ///
    /// # Panics
    /// Panics if the length of the string is greater than `u32::max_value()`
    #[inline(always)]
    pub fn of_str(s: &str) -> TextUnit {
        if s.len() > u32::max_value() as usize {
            panic!("string is to long")
        }
        TextUnit(s.len() as u32)
    }

    #[inline(always)]
    pub fn checked_sub(self, other: TextUnit) -> Option<TextUnit> {
        self.0.checked_sub(other.0).map(TextUnit)
    }

    #[inline(always)]
    pub fn from_usize(size: usize) -> TextUnit {
        #[cfg(debug_assertions)]
        {
            if size > u32::max_value() as usize {
                panic!("overflow when converting to TextUnit: {}", size)
            }
        }
        (size as u32).into()
    }

    #[inline(always)]
    pub fn to_usize(self) -> usize {
        u32::from(self) as usize
    }
}

impl fmt::Debug for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for TextUnit {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<TextUnit> for u32 {
    #[inline(always)]
    fn from(tu: TextUnit) -> u32 {
        tu.0
    }
}

impl From<u32> for TextUnit {
    #[inline(always)]
    fn from(tu: u32) -> TextUnit {
        TextUnit(tu)
    }
}

macro_rules! unit_ops_impls {
    ($T:ident, $f:ident, $op:tt, $AT:ident, $af:ident) => {

impl ops::$T<TextUnit> for TextUnit {
    type Output = TextUnit;
    #[inline(always)]
    fn $f(self, rhs: TextUnit) -> TextUnit {
        TextUnit(self.0 $op rhs.0)
    }
}

impl<'a> ops::$T<&'a TextUnit> for TextUnit {
    type Output = TextUnit;
    #[inline(always)]
    fn $f(self, rhs: &'a TextUnit) -> TextUnit {
        ops::$T::$f(self, *rhs)
    }
}

impl<'a> ops::$T<TextUnit> for &'a TextUnit {
    type Output = TextUnit;
    #[inline(always)]
    fn $f(self, rhs: TextUnit) -> TextUnit {
        ops::$T::$f(*self, rhs)
    }
}

impl<'a, 'b> ops::$T<&'a TextUnit> for &'b TextUnit {
    type Output = TextUnit;
    #[inline(always)]
    fn $f(self, rhs: &'a TextUnit) -> TextUnit {
        ops::$T::$f(*self, *rhs)
    }
}

impl ops::$AT<TextUnit> for TextUnit {
    #[inline(always)]
    fn $af(&mut self, rhs: TextUnit) {
        self.0 = self.0 $op rhs.0
    }
}

impl<'a> ops::$AT<&'a TextUnit> for TextUnit {
    #[inline(always)]
    fn $af(&mut self, rhs: &'a TextUnit) {
        ops::$AT::$af(self, *rhs)
    }
}
    };
}

macro_rules! range_ops_impls {
    ($T:ident, $f:ident, $op:tt, $AT:ident, $af:ident) => {

impl ops::$T<TextUnit> for TextRange {
    type Output = TextRange;
    #[inline(always)]
    fn $f(self, rhs: TextUnit) -> TextRange {
        TextRange::from_to(
            self.start() $op rhs,
            self.end() $op rhs,
        )
    }
}

impl<'a> ops::$T<&'a TextUnit> for TextRange {
    type Output = TextRange;
    #[inline(always)]
    fn $f(self, rhs: &'a TextUnit) -> TextRange {
        TextRange::from_to(
            self.start() $op rhs,
            self.end() $op rhs,
        )
    }
}

impl<'a> ops::$T<TextUnit> for &'a TextRange {
    type Output = TextRange;
    #[inline(always)]
    fn $f(self, rhs: TextUnit) -> TextRange {
        TextRange::from_to(
            self.start() $op rhs,
            self.end() $op rhs,
        )
    }
}

impl<'a, 'b> ops::$T<&'a TextUnit> for &'b TextRange {
    type Output = TextRange;
    #[inline(always)]
    fn $f(self, rhs: &'a TextUnit) -> TextRange {
        TextRange::from_to(
            self.start() $op rhs,
            self.end() $op rhs,
        )
    }
}

impl ops::$AT<TextUnit> for TextRange {
    #[inline(always)]
    fn $af(&mut self, rhs: TextUnit) {
        *self = *self $op rhs
    }
}

impl<'a> ops::$AT<&'a TextUnit> for TextRange {
    #[inline(always)]
    fn $af(&mut self, rhs: &'a TextUnit) {
        *self = *self $op rhs
    }
}
    };
}

unit_ops_impls!(Add, add, +, AddAssign, add_assign);
unit_ops_impls!(Sub, sub, -, SubAssign, sub_assign);
range_ops_impls!(Add, add, +, AddAssign, add_assign);
range_ops_impls!(Sub, sub, -, SubAssign, sub_assign);

impl<'a> iter::Sum<&'a TextUnit> for TextUnit {
    fn sum<I: Iterator<Item = &'a TextUnit>>(iter: I) -> TextUnit {
        iter.fold(TextUnit::from(0), ops::Add::add)
    }
}

impl iter::Sum<TextUnit> for TextUnit {
    fn sum<I: Iterator<Item = TextUnit>>(iter: I) -> TextUnit {
        iter.fold(TextUnit::from(0), ops::Add::add)
    }
}

/// A range in the text, represented as a pair of `TextUnit`s.
///
/// # Panics
/// Slicing a `&str` with `TextRange` panics if the result is
/// not a valid utf8 string.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextRange {
    start: TextUnit,
    end: TextUnit,
}

impl fmt::Debug for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for TextRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}; {})", self.start(), self.end())
    }
}

impl TextRange {
    // BREAK: TextRange::new(from..to)?
    // BREAK: TextRange(from, to)?
    /// The left-inclusive range (`[from..to)`) between to points in the text
    #[inline(always)]
    pub fn from_to(from: TextUnit, to: TextUnit) -> TextRange {
        assert!(from <= to, "Invalid text range [{}; {})", from, to);
        TextRange {
            start: from,
            end: to,
        }
    }

    /// The left-inclusive range (`[offset..offset + len)`) between to points in the text
    #[inline(always)]
    pub fn offset_len(offset: TextUnit, len: TextUnit) -> TextRange {
        TextRange::from_to(offset, offset + len)
    }

    // BREAK: pass by value
    /// The inclusive start of this range
    #[inline(always)]
    pub fn start(&self) -> TextUnit {
        self.start
    }

    // BREAK: pass by value
    /// The exclusive end of this range
    #[inline(always)]
    pub fn end(&self) -> TextUnit {
        self.end
    }

    // BREAK: pass by value
    /// The length of this range
    #[inline(always)]
    pub fn len(&self) -> TextUnit {
        self.end - self.start
    }

    // BREAK: pass by value
    /// Is this range empty of any content?
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.start() == self.end()
    }

    // BREAK: pass by value
    #[inline(always)]
    pub fn is_subrange(&self, other: &TextRange) -> bool {
        other.start() <= self.start() && self.end() <= other.end()
    }

    // BREAK: pass by value
    #[inline(always)]
    pub fn intersection(&self, other: &TextRange) -> Option<TextRange> {
        let start = self.start.max(other.start());
        let end = self.end.min(other.end());
        if start <= end {
            Some(TextRange::from_to(start, end))
        } else {
            None
        }
    }

    // BREAK: pass by value
    #[inline(always)]
    /// The smallest convex set that contains both ranges
    pub fn convex_hull(&self, other: &TextRange) -> TextRange {
        let start = self.start().min(other.start());
        let end = self.end().max(other.end());
        TextRange::from_to(start, end)
    }

    // BREAK: pass by value
    #[inline(always)]
    pub fn contains(&self, offset: TextUnit) -> bool {
        self.start() <= offset && offset < self.end()
    }

    // BREAK: pass by value
    #[inline(always)]
    pub fn contains_inclusive(&self, offset: TextUnit) -> bool {
        self.start() <= offset && offset <= self.end()
    }

    #[inline(always)]
    pub fn checked_sub(self, other: TextUnit) -> Option<TextRange> {
        let res = TextRange::offset_len(
            self.start().checked_sub(other)?,
            self.len()
        );
        Some(res)
    }
}

impl ops::RangeBounds<TextUnit> for TextRange {
    fn start_bound(&self) -> ops::Bound<&TextUnit> {
        ops::Bound::Included(&self.start)
    }

    fn end_bound(&self) -> ops::Bound<&TextUnit> {
        ops::Bound::Excluded(&self.end)
    }
}

impl ops::Index<TextRange> for str {
    type Output = str;

    fn index(&self, index: TextRange) -> &str {
        &self[index.start().0 as usize..index.end().0 as usize]
    }
}

impl ops::Index<TextRange> for String {
    type Output = str;

    fn index(&self, index: TextRange) -> &str {
        &self.as_str()[index]
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use {TextRange, TextUnit};

    impl Serialize for TextUnit {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for TextUnit {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let value = Deserialize::deserialize(deserializer)?;
            Ok(TextUnit(value))
        }
    }

    impl Serialize for TextRange {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            (self.start, self.end).serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for TextRange {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let (start, end) = Deserialize::deserialize(deserializer)?;
            Ok(TextRange { start, end })
        }
    }
}

#[cfg(feature = "deepsize")]
mod deepsize_impls {
    deepsize::known_deep_size!(0, crate::TextUnit, crate::TextRange);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(from: u32, to: u32) -> TextRange {
        TextRange::from_to(from.into(), to.into())
    }

    #[test]
    fn test_sum() {
        let xs: Vec<TextUnit> = vec![0.into(), 1.into(), 2.into()];
        assert_eq!(xs.iter().sum::<TextUnit>(), 3.into());
        assert_eq!(xs.into_iter().sum::<TextUnit>(), 3.into());
    }

    #[test]
    fn test_ops() {
        let range = r(10, 20);
        let u: TextUnit = 5.into();
        assert_eq!(range + u, r(15, 25));
        assert_eq!(range - u, r(5, 15));
    }

    #[test]
    fn test_checked_ops() {
        let x: TextUnit = 1.into();
        assert_eq!(x.checked_sub(1.into()), Some(0.into()));
        assert_eq!(x.checked_sub(2.into()), None);

        assert_eq!(r(1, 2).checked_sub(1.into()), Some(r(0, 1)));
        assert_eq!(x.checked_sub(2.into()), None);
    }

    #[test]
    fn test_subrange() {
        let r1 = r(2, 4);
        let r2 = r(2, 3);
        let r3 = r(1, 3);
        assert!(r2.is_subrange(&r1));
        assert!(!r3.is_subrange(&r1));
    }

    #[test]
    fn check_intersection() {
        assert_eq!(r(1, 2).intersection(&r(2, 3)), Some(r(2, 2)));
        assert_eq!(r(1, 5).intersection(&r(2, 3)), Some(r(2, 3)));
        assert_eq!(r(1, 2).intersection(&r(3, 4)), None);
    }

    #[test]
    fn check_convex_hull() {
        assert_eq!(r(1, 2).convex_hull(&r(2, 3)), r(1, 3));
        assert_eq!(r(1, 5).convex_hull(&r(2, 3)), r(1, 5));
        assert_eq!(r(1, 2).convex_hull(&r(4, 5)), r(1, 5));
    }

    #[test]
    fn check_contains() {
        assert!(!r(1, 3).contains(0.into()));
        assert!(r(1, 3).contains(1.into()));
        assert!(r(1, 3).contains(2.into()));
        assert!(!r(1, 3).contains(3.into()));
        assert!(!r(1, 3).contains(4.into()));

        assert!(!r(1, 3).contains_inclusive(0.into()));
        assert!(r(1, 3).contains_inclusive(1.into()));
        assert!(r(1, 3).contains_inclusive(2.into()));
        assert!(r(1, 3).contains_inclusive(3.into()));
        assert!(!r(1, 3).contains_inclusive(4.into()));
    }
}
