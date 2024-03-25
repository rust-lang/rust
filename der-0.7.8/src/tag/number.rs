//! ASN.1 tag numbers

use super::Tag;
use crate::{Error, ErrorKind, Result};
use core::fmt;

/// ASN.1 tag numbers (i.e. lower 5 bits of a [`Tag`]).
///
/// From X.690 Section 8.1.2.2:
///
/// > bits 5 to 1 shall encode the number of the tag as a binary integer with
/// > bit 5 as the most significant bit.
///
/// This library supports tag numbers ranging from zero to 30 (inclusive),
/// which can be represented as a single identifier octet.
///
/// Section 8.1.2.4 describes how to support multi-byte tag numbers, which are
/// encoded by using a leading tag number of 31 (`0b11111`). This library
/// deliberately does not support this: tag numbers greater than 30 are
/// disallowed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct TagNumber(pub(super) u8);

impl TagNumber {
    /// Tag number `0`
    pub const N0: Self = Self(0);

    /// Tag number `1`
    pub const N1: Self = Self(1);

    /// Tag number `2`
    pub const N2: Self = Self(2);

    /// Tag number `3`
    pub const N3: Self = Self(3);

    /// Tag number `4`
    pub const N4: Self = Self(4);

    /// Tag number `5`
    pub const N5: Self = Self(5);

    /// Tag number `6`
    pub const N6: Self = Self(6);

    /// Tag number `7`
    pub const N7: Self = Self(7);

    /// Tag number `8`
    pub const N8: Self = Self(8);

    /// Tag number `9`
    pub const N9: Self = Self(9);

    /// Tag number `10`
    pub const N10: Self = Self(10);

    /// Tag number `11`
    pub const N11: Self = Self(11);

    /// Tag number `12`
    pub const N12: Self = Self(12);

    /// Tag number `13`
    pub const N13: Self = Self(13);

    /// Tag number `14`
    pub const N14: Self = Self(14);

    /// Tag number `15`
    pub const N15: Self = Self(15);

    /// Tag number `16`
    pub const N16: Self = Self(16);

    /// Tag number `17`
    pub const N17: Self = Self(17);

    /// Tag number `18`
    pub const N18: Self = Self(18);

    /// Tag number `19`
    pub const N19: Self = Self(19);

    /// Tag number `20`
    pub const N20: Self = Self(20);

    /// Tag number `21`
    pub const N21: Self = Self(21);

    /// Tag number `22`
    pub const N22: Self = Self(22);

    /// Tag number `23`
    pub const N23: Self = Self(23);

    /// Tag number `24`
    pub const N24: Self = Self(24);

    /// Tag number `25`
    pub const N25: Self = Self(25);

    /// Tag number `26`
    pub const N26: Self = Self(26);

    /// Tag number `27`
    pub const N27: Self = Self(27);

    /// Tag number `28`
    pub const N28: Self = Self(28);

    /// Tag number `29`
    pub const N29: Self = Self(29);

    /// Tag number `30`
    pub const N30: Self = Self(30);

    /// Mask value used to obtain the tag number from a tag octet.
    pub(super) const MASK: u8 = 0b11111;

    /// Maximum tag number supported (inclusive).
    const MAX: u8 = 30;

    /// Create a new tag number (const-friendly).
    ///
    /// Panics if the tag number is greater than `30`.
    /// For a fallible conversion, use [`TryFrom`] instead.
    pub const fn new(byte: u8) -> Self {
        #[allow(clippy::panic)]
        if byte > Self::MAX {
            panic!("tag number out of range");
        }

        Self(byte)
    }

    /// Create an `APPLICATION` tag with this tag number.
    pub fn application(self, constructed: bool) -> Tag {
        Tag::Application {
            constructed,
            number: self,
        }
    }

    /// Create a `CONTEXT-SPECIFIC` tag with this tag number.
    pub fn context_specific(self, constructed: bool) -> Tag {
        Tag::ContextSpecific {
            constructed,
            number: self,
        }
    }

    /// Create a `PRIVATE` tag with this tag number.
    pub fn private(self, constructed: bool) -> Tag {
        Tag::Private {
            constructed,
            number: self,
        }
    }

    /// Get the inner value.
    pub fn value(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for TagNumber {
    type Error = Error;

    fn try_from(byte: u8) -> Result<Self> {
        match byte {
            0..=Self::MAX => Ok(Self(byte)),
            _ => Err(ErrorKind::TagNumberInvalid.into()),
        }
    }
}

impl From<TagNumber> for u8 {
    fn from(tag_number: TagNumber) -> u8 {
        tag_number.0
    }
}

impl fmt::Display for TagNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// Implement by hand because the derive would create invalid values.
// Use the constructor to create a valid value.
#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for TagNumber {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self::new(u.int_in_range(0..=Self::MAX)?))
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        u8::size_hint(depth)
    }
}
