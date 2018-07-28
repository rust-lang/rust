#[cfg(feature = "serde")]
extern crate serde;

use std::{fmt, ops, iter};


/// An offset into text.
/// Offset is represented as `u32` storing number of utf8-bytes,
/// but most of the clients should treat it like opaque measure.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TextUnit(u32);

impl TextUnit {
    /// `TextUnit` equal to the length of this char.
    pub fn of_char(c: char) -> TextUnit {
        TextUnit(c.len_utf8() as u32)
    }

    /// `TextUnit` equal to the length of this string.
    ///
    /// # Panics
    /// Panics if the length of the string is greater than `u32::max_value()`
    pub fn of_str(s: &str) -> TextUnit {
        if s.len() > u32::max_value() as usize {
            panic!("string is to long")
        }
        TextUnit(s.len() as u32)
    }
}

impl fmt::Debug for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl fmt::Display for TextUnit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<TextUnit> for u32 {
    fn from(tu: TextUnit) -> u32 {
        tu.0
    }
}

impl From<u32> for TextUnit {
    fn from(tu: u32) -> TextUnit {
        TextUnit(tu)
    }
}

macro_rules! ops_impls {
    ($T:ident, $f:ident, $op:tt, $AT:ident, $af:ident) => {

impl ops::$T<TextUnit> for TextUnit {
    type Output = TextUnit;
    fn $f(self, rhs: TextUnit) -> TextUnit {
        TextUnit(self.0 $op rhs.0)
    }
}

impl<'a> ops::$T<&'a TextUnit> for TextUnit {
    type Output = TextUnit;
    fn $f(self, rhs: &'a TextUnit) -> TextUnit {
        ops::$T::$f(self, *rhs)
    }
}

impl<'a> ops::$T<TextUnit> for &'a TextUnit {
    type Output = TextUnit;
    fn $f(self, rhs: TextUnit) -> TextUnit {
        ops::$T::$f(*self, rhs)
    }
}

impl<'a, 'b> ops::$T<&'a TextUnit> for &'b TextUnit {
    type Output = TextUnit;
    fn $f(self, rhs: &'a TextUnit) -> TextUnit {
        ops::$T::$f(*self, *rhs)
    }
}

impl ops::$AT<TextUnit> for TextUnit {
    fn $af(&mut self, rhs: TextUnit) {
        self.0 = self.0 $op rhs.0
    }
}

impl<'a> ops::$AT<&'a TextUnit> for TextUnit {
    fn $af(&mut self, rhs: &'a TextUnit) {
        ops::$AT::$af(self, *rhs)
    }
}
    };
}

ops_impls!(Add, add, +, AddAssign, add_assign);
ops_impls!(Sub, sub, -, SubAssign, sub_assign);

impl<'a> iter::Sum<&'a TextUnit> for TextUnit {
    fn sum<I: Iterator<Item=&'a TextUnit>>(iter: I) -> TextUnit {
        iter.fold(TextUnit::from(0), ops::Add::add)
    }
}

impl iter::Sum<TextUnit> for TextUnit {
    fn sum<I: Iterator<Item=TextUnit>>(iter: I) -> TextUnit {
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
    /// The left-inclusive range (`[from..to)`) between to points in the text
    pub fn from_to(from: TextUnit, to: TextUnit) -> TextRange {
        assert!(from <= to, "Invalid text range [{}; {})", from, to);
        TextRange {
            start: from,
            end: to,
        }
    }

    /// The left-inclusive range (`[offset..offset + len)`) between to points in the text
    pub fn offset_len(offset: TextUnit, len: TextUnit) -> TextRange {
        TextRange::from_to(offset, offset + len)
    }

    /// The inclusive start of this range
    pub fn start(&self) -> TextUnit {
        self.start
    }

    /// The exclusive end of this range
    pub fn end(&self) -> TextUnit {
        self.end
    }

    /// The length of this range
    pub fn len(&self) -> TextUnit {
        self.end - self.start
    }

    /// Is this range empty of any content?
    pub fn is_empty(&self) -> bool {
        self.start() == self.end()
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
    use serde::{Serialize, Serializer, Deserialize, Deserializer};
    use {TextUnit, TextRange};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let xs: Vec<TextUnit> = vec![0.into(), 1.into(), 2.into()];
        assert_eq!(xs.iter().sum::<TextUnit>(), 3.into());
        assert_eq!(xs.into_iter().sum::<TextUnit>(), 3.into());
    }
}
