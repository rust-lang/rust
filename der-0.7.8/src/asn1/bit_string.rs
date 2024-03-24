//! ASN.1 `BIT STRING` support.

use crate::{
    BytesRef, DecodeValue, DerOrd, EncodeValue, Error, ErrorKind, FixedTag, Header, Length, Reader,
    Result, Tag, ValueOrd, Writer,
};
use core::{cmp::Ordering, iter::FusedIterator};

/// ASN.1 `BIT STRING` type.
///
/// This type contains a sequence of any number of bits, modeled internally as
/// a sequence of bytes with a known number of "unused bits".
///
/// This is a zero-copy reference type which borrows from the input data.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct BitStringRef<'a> {
    /// Number of unused bits in the final octet.
    unused_bits: u8,

    /// Length of this `BIT STRING` in bits.
    bit_length: usize,

    /// Bitstring represented as a slice of bytes.
    inner: BytesRef<'a>,
}

impl<'a> BitStringRef<'a> {
    /// Maximum number of unused bits allowed.
    pub const MAX_UNUSED_BITS: u8 = 7;

    /// Create a new ASN.1 `BIT STRING` from a byte slice.
    ///
    /// Accepts an optional number of "unused bits" (0-7) which are omitted
    /// from the final octet. This number is 0 if the value is octet-aligned.
    pub fn new(unused_bits: u8, bytes: &'a [u8]) -> Result<Self> {
        if (unused_bits > Self::MAX_UNUSED_BITS) || (unused_bits != 0 && bytes.is_empty()) {
            return Err(Self::TAG.value_error());
        }

        let inner = BytesRef::new(bytes).map_err(|_| Self::TAG.length_error())?;

        let bit_length = usize::try_from(inner.len())?
            .checked_mul(8)
            .and_then(|n| n.checked_sub(usize::from(unused_bits)))
            .ok_or(ErrorKind::Overflow)?;

        Ok(Self {
            unused_bits,
            bit_length,
            inner,
        })
    }

    /// Create a new ASN.1 `BIT STRING` from the given bytes.
    ///
    /// The "unused bits" are set to 0.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        Self::new(0, bytes)
    }

    /// Get the number of unused bits in this byte slice.
    pub fn unused_bits(&self) -> u8 {
        self.unused_bits
    }

    /// Is the number of unused bits a value other than 0?
    pub fn has_unused_bits(&self) -> bool {
        self.unused_bits != 0
    }

    /// Get the length of this `BIT STRING` in bits.
    pub fn bit_len(&self) -> usize {
        self.bit_length
    }

    /// Get the number of bytes/octets needed to represent this `BIT STRING`
    /// when serialized in an octet-aligned manner.
    pub fn byte_len(&self) -> Length {
        self.inner.len()
    }

    /// Is the inner byte slice empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrow the inner byte slice.
    ///
    /// Returns `None` if the number of unused bits is *not* equal to zero,
    /// i.e. if the `BIT STRING` is not octet aligned.
    ///
    /// Use [`BitString::raw_bytes`] to obtain access to the raw value
    /// regardless of the presence of unused bits.
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        if self.has_unused_bits() {
            None
        } else {
            Some(self.raw_bytes())
        }
    }

    /// Borrow the raw bytes of this `BIT STRING`.
    ///
    /// Note that the byte string may contain extra unused bits in the final
    /// octet. If the number of unused bits is expected to be 0, the
    /// [`BitStringRef::as_bytes`] function can be used instead.
    pub fn raw_bytes(&self) -> &'a [u8] {
        self.inner.as_slice()
    }

    /// Iterator over the bits of this `BIT STRING`.
    pub fn bits(self) -> BitStringIter<'a> {
        BitStringIter {
            bit_string: self,
            position: 0,
        }
    }
}

impl_any_conversions!(BitStringRef<'a>, 'a);

impl<'a> DecodeValue<'a> for BitStringRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        let header = Header {
            tag: header.tag,
            length: (header.length - Length::ONE)?,
        };

        let unused_bits = reader.read_byte()?;
        let inner = BytesRef::decode_value(reader, header)?;
        Self::new(unused_bits, inner.as_slice())
    }
}

impl EncodeValue for BitStringRef<'_> {
    fn value_len(&self) -> Result<Length> {
        self.byte_len() + Length::ONE
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write_byte(self.unused_bits)?;
        writer.write(self.raw_bytes())
    }
}

impl ValueOrd for BitStringRef<'_> {
    fn value_cmp(&self, other: &Self) -> Result<Ordering> {
        match self.unused_bits.cmp(&other.unused_bits) {
            Ordering::Equal => self.inner.der_cmp(&other.inner),
            ordering => Ok(ordering),
        }
    }
}

impl<'a> From<&BitStringRef<'a>> for BitStringRef<'a> {
    fn from(value: &BitStringRef<'a>) -> BitStringRef<'a> {
        *value
    }
}

impl<'a> TryFrom<&'a [u8]> for BitStringRef<'a> {
    type Error = Error;

    fn try_from(bytes: &'a [u8]) -> Result<BitStringRef<'a>> {
        BitStringRef::from_bytes(bytes)
    }
}

/// Hack for simplifying the custom derive use case.
impl<'a> TryFrom<&&'a [u8]> for BitStringRef<'a> {
    type Error = Error;

    fn try_from(bytes: &&'a [u8]) -> Result<BitStringRef<'a>> {
        BitStringRef::from_bytes(bytes)
    }
}

impl<'a> TryFrom<BitStringRef<'a>> for &'a [u8] {
    type Error = Error;

    fn try_from(bit_string: BitStringRef<'a>) -> Result<&'a [u8]> {
        bit_string
            .as_bytes()
            .ok_or_else(|| Tag::BitString.value_error())
    }
}

impl<'a> FixedTag for BitStringRef<'a> {
    const TAG: Tag = Tag::BitString;
}

// Implement by hand because the derive would create invalid values.
// Use the constructor to create a valid value.
#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for BitStringRef<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Self::new(
            u.int_in_range(0..=Self::MAX_UNUSED_BITS)?,
            BytesRef::arbitrary(u)?.as_slice(),
        )
        .map_err(|_| arbitrary::Error::IncorrectFormat)
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        arbitrary::size_hint::and(u8::size_hint(depth), BytesRef::size_hint(depth))
    }
}

#[cfg(feature = "alloc")]
pub use self::allocating::BitString;

#[cfg(feature = "alloc")]
mod allocating {
    use super::*;
    use crate::referenced::*;
    use alloc::vec::Vec;

    /// Owned form of ASN.1 `BIT STRING` type.
    ///
    /// This type provides the same functionality as [`BitStringRef`] but owns the
    /// backing data.
    #[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
    pub struct BitString {
        /// Number of unused bits in the final octet.
        unused_bits: u8,

        /// Length of this `BIT STRING` in bits.
        bit_length: usize,

        /// Bitstring represented as a slice of bytes.
        inner: Vec<u8>,
    }

    impl BitString {
        /// Maximum number of unused bits allowed.
        pub const MAX_UNUSED_BITS: u8 = 7;

        /// Create a new ASN.1 `BIT STRING` from a byte slice.
        ///
        /// Accepts an optional number of "unused bits" (0-7) which are omitted
        /// from the final octet. This number is 0 if the value is octet-aligned.
        pub fn new(unused_bits: u8, bytes: impl Into<Vec<u8>>) -> Result<Self> {
            let inner = bytes.into();

            // Ensure parameters parse successfully as a `BitStringRef`.
            let bit_length = BitStringRef::new(unused_bits, &inner)?.bit_length;

            Ok(BitString {
                unused_bits,
                bit_length,
                inner,
            })
        }

        /// Create a new ASN.1 `BIT STRING` from the given bytes.
        ///
        /// The "unused bits" are set to 0.
        pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
            Self::new(0, bytes)
        }

        /// Get the number of unused bits in the octet serialization of this
        /// `BIT STRING`.
        pub fn unused_bits(&self) -> u8 {
            self.unused_bits
        }

        /// Is the number of unused bits a value other than 0?
        pub fn has_unused_bits(&self) -> bool {
            self.unused_bits != 0
        }

        /// Get the length of this `BIT STRING` in bits.
        pub fn bit_len(&self) -> usize {
            self.bit_length
        }

        /// Is the inner byte slice empty?
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Borrow the inner byte slice.
        ///
        /// Returns `None` if the number of unused bits is *not* equal to zero,
        /// i.e. if the `BIT STRING` is not octet aligned.
        ///
        /// Use [`BitString::raw_bytes`] to obtain access to the raw value
        /// regardless of the presence of unused bits.
        pub fn as_bytes(&self) -> Option<&[u8]> {
            if self.has_unused_bits() {
                None
            } else {
                Some(self.raw_bytes())
            }
        }

        /// Borrow the raw bytes of this `BIT STRING`.
        pub fn raw_bytes(&self) -> &[u8] {
            self.inner.as_slice()
        }

        /// Iterator over the bits of this `BIT STRING`.
        pub fn bits(&self) -> BitStringIter<'_> {
            BitStringRef::from(self).bits()
        }
    }

    impl_any_conversions!(BitString);

    impl<'a> DecodeValue<'a> for BitString {
        fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
            let inner_len = (header.length - Length::ONE)?;
            let unused_bits = reader.read_byte()?;
            let inner = reader.read_vec(inner_len)?;
            Self::new(unused_bits, inner)
        }
    }

    impl EncodeValue for BitString {
        fn value_len(&self) -> Result<Length> {
            Length::ONE + Length::try_from(self.inner.len())?
        }

        fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
            writer.write_byte(self.unused_bits)?;
            writer.write(&self.inner)
        }
    }

    impl FixedTag for BitString {
        const TAG: Tag = Tag::BitString;
    }

    impl<'a> From<&'a BitString> for BitStringRef<'a> {
        fn from(bit_string: &'a BitString) -> BitStringRef<'a> {
            // Ensured to parse successfully in constructor
            BitStringRef::new(bit_string.unused_bits, &bit_string.inner)
                .expect("invalid BIT STRING")
        }
    }

    impl ValueOrd for BitString {
        fn value_cmp(&self, other: &Self) -> Result<Ordering> {
            match self.unused_bits.cmp(&other.unused_bits) {
                Ordering::Equal => self.inner.der_cmp(&other.inner),
                ordering => Ok(ordering),
            }
        }
    }

    // Implement by hand because the derive would create invalid values.
    // Use the constructor to create a valid value.
    #[cfg(feature = "arbitrary")]
    impl<'a> arbitrary::Arbitrary<'a> for BitString {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            Self::new(
                u.int_in_range(0..=Self::MAX_UNUSED_BITS)?,
                BytesRef::arbitrary(u)?.as_slice(),
            )
            .map_err(|_| arbitrary::Error::IncorrectFormat)
        }

        fn size_hint(depth: usize) -> (usize, Option<usize>) {
            arbitrary::size_hint::and(u8::size_hint(depth), BytesRef::size_hint(depth))
        }
    }

    impl<'a> RefToOwned<'a> for BitStringRef<'a> {
        type Owned = BitString;
        fn ref_to_owned(&self) -> Self::Owned {
            BitString {
                unused_bits: self.unused_bits,
                bit_length: self.bit_length,
                inner: Vec::from(self.inner.as_slice()),
            }
        }
    }

    impl OwnedToRef for BitString {
        type Borrowed<'a> = BitStringRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            self.into()
        }
    }
}

/// Iterator over the bits of a [`BitString`].
pub struct BitStringIter<'a> {
    /// [`BitString`] being iterated over.
    bit_string: BitStringRef<'a>,

    /// Current bit position within the iterator.
    position: usize,
}

impl<'a> Iterator for BitStringIter<'a> {
    type Item = bool;

    #[allow(clippy::integer_arithmetic)]
    fn next(&mut self) -> Option<bool> {
        if self.position >= self.bit_string.bit_len() {
            return None;
        }

        let byte = self.bit_string.raw_bytes().get(self.position / 8)?;
        let bit = 1u8 << (7 - (self.position % 8));
        self.position = self.position.checked_add(1)?;
        Some(byte & bit != 0)
    }
}

impl<'a> ExactSizeIterator for BitStringIter<'a> {
    fn len(&self) -> usize {
        self.bit_string.bit_len()
    }
}

impl<'a> FusedIterator for BitStringIter<'a> {}

#[cfg(feature = "flagset")]
impl<T: flagset::Flags> FixedTag for flagset::FlagSet<T> {
    const TAG: Tag = BitStringRef::TAG;
}

#[cfg(feature = "flagset")]
impl<T> ValueOrd for flagset::FlagSet<T>
where
    T: flagset::Flags,
    T::Type: Ord,
{
    fn value_cmp(&self, other: &Self) -> Result<Ordering> {
        Ok(self.bits().cmp(&other.bits()))
    }
}

#[cfg(feature = "flagset")]
#[allow(clippy::integer_arithmetic)]
impl<'a, T> DecodeValue<'a> for flagset::FlagSet<T>
where
    T: flagset::Flags,
    T::Type: From<bool>,
    T::Type: core::ops::Shl<usize, Output = T::Type>,
{
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        let position = reader.position();
        let bits = BitStringRef::decode_value(reader, header)?;

        let mut flags = T::none().bits();

        if bits.bit_len() > core::mem::size_of_val(&flags) * 8 {
            return Err(Error::new(ErrorKind::Overlength, position));
        }

        for (i, bit) in bits.bits().enumerate() {
            flags |= T::Type::from(bit) << i;
        }

        Ok(Self::new_truncated(flags))
    }
}

#[cfg(feature = "flagset")]
#[allow(clippy::integer_arithmetic)]
#[inline(always)]
fn encode_flagset<T>(set: &flagset::FlagSet<T>) -> (usize, [u8; 16])
where
    T: flagset::Flags,
    u128: From<T::Type>,
{
    let bits: u128 = set.bits().into();
    let mut swap = 0u128;

    for i in 0..128 {
        let on = bits & (1 << i);
        swap |= on >> i << (128 - i - 1);
    }

    (bits.leading_zeros() as usize, swap.to_be_bytes())
}

#[cfg(feature = "flagset")]
#[allow(clippy::cast_possible_truncation, clippy::integer_arithmetic)]
impl<T: flagset::Flags> EncodeValue for flagset::FlagSet<T>
where
    T::Type: From<bool>,
    T::Type: core::ops::Shl<usize, Output = T::Type>,
    u128: From<T::Type>,
{
    fn value_len(&self) -> Result<Length> {
        let (lead, buff) = encode_flagset(self);
        let buff = &buff[..buff.len() - lead / 8];
        BitStringRef::new((lead % 8) as u8, buff)?.value_len()
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        let (lead, buff) = encode_flagset(self);
        let buff = &buff[..buff.len() - lead / 8];
        BitStringRef::new((lead % 8) as u8, buff)?.encode_value(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitStringRef, Result, Tag};
    use crate::asn1::AnyRef;
    use hex_literal::hex;

    /// Parse a `BitString` from an ASN.1 `Any` value to test decoding behaviors.
    fn parse_bitstring(bytes: &[u8]) -> Result<BitStringRef<'_>> {
        AnyRef::new(Tag::BitString, bytes)?.try_into()
    }

    #[test]
    fn decode_empty_bitstring() {
        let bs = parse_bitstring(&hex!("00")).unwrap();
        assert_eq!(bs.as_bytes().unwrap(), &[]);
    }

    #[test]
    fn decode_non_empty_bitstring() {
        let bs = parse_bitstring(&hex!("00010203")).unwrap();
        assert_eq!(bs.as_bytes().unwrap(), &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn decode_bitstring_with_unused_bits() {
        let bs = parse_bitstring(&hex!("066e5dc0")).unwrap();
        assert_eq!(bs.unused_bits(), 6);
        assert_eq!(bs.raw_bytes(), &hex!("6e5dc0"));

        // Expected: 011011100101110111
        let mut bits = bs.bits();
        assert_eq!(bits.len(), 18);

        for bit in [0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1] {
            assert_eq!(u8::from(bits.next().unwrap()), bit)
        }

        // Ensure `None` is returned on successive calls
        assert_eq!(bits.next(), None);
        assert_eq!(bits.next(), None);
    }

    #[test]
    fn reject_unused_bits_in_empty_string() {
        assert_eq!(
            parse_bitstring(&[0x03]).err().unwrap().kind(),
            Tag::BitString.value_error().kind()
        )
    }
}
