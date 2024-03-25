//! Unsigned integer decoders/encoders.

use super::value_cmp;
use crate::{
    ord::OrdIsValueOrd, AnyRef, BytesRef, DecodeValue, EncodeValue, Error, ErrorKind, FixedTag,
    Header, Length, Reader, Result, Tag, ValueOrd, Writer,
};
use core::cmp::Ordering;

#[cfg(feature = "alloc")]
pub use allocating::Uint;

macro_rules! impl_encoding_traits {
    ($($uint:ty),+) => {
        $(
            impl<'a> DecodeValue<'a> for $uint {
                fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
                    // Integers always encodes as a signed value, unsigned gets a leading 0x00 that
                    // needs to be stripped off. We need to provide room for it.
                    const UNSIGNED_HEADROOM: usize = 1;

                    let mut buf = [0u8; (Self::BITS as usize / 8) + UNSIGNED_HEADROOM];
                    let max_length = u32::from(header.length) as usize;

                    if max_length > buf.len() {
                        return Err(Self::TAG.non_canonical_error());
                    }

                    let bytes = reader.read_into(&mut buf[..max_length])?;

                    let result = Self::from_be_bytes(decode_to_array(bytes)?);

                    // Ensure we compute the same encoded length as the original any value
                    if header.length != result.value_len()? {
                        return Err(Self::TAG.non_canonical_error());
                    }

                    Ok(result)
                }
            }

            impl EncodeValue for $uint {
                fn value_len(&self) -> Result<Length> {
                    encoded_len(&self.to_be_bytes())
                }

                fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
                    encode_bytes(writer, &self.to_be_bytes())
                }
            }

            impl FixedTag for $uint {
                const TAG: Tag = Tag::Integer;
            }

            impl ValueOrd for $uint {
                fn value_cmp(&self, other: &Self) -> Result<Ordering> {
                    value_cmp(*self, *other)
                }
            }

            impl TryFrom<AnyRef<'_>> for $uint {
                type Error = Error;

                fn try_from(any: AnyRef<'_>) -> Result<Self> {
                    any.decode_as()
                }
            }
        )+
    };
}

impl_encoding_traits!(u8, u16, u32, u64, u128);

/// Unsigned arbitrary precision ASN.1 `INTEGER` reference type.
///
/// Provides direct access to the underlying big endian bytes which comprise an
/// unsigned integer value.
///
/// Intended for use cases like very large integers that are used in
/// cryptographic applications (e.g. keys, signatures).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct UintRef<'a> {
    /// Inner value
    inner: BytesRef<'a>,
}

impl<'a> UintRef<'a> {
    /// Create a new [`UintRef`] from a byte slice.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let inner = BytesRef::new(strip_leading_zeroes(bytes))
            .map_err(|_| ErrorKind::Length { tag: Self::TAG })?;

        Ok(Self { inner })
    }

    /// Borrow the inner byte slice which contains the least significant bytes
    /// of a big endian integer value with all leading zeros stripped.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.inner.as_slice()
    }

    /// Get the length of this [`UintRef`] in bytes.
    pub fn len(&self) -> Length {
        self.inner.len()
    }

    /// Is the inner byte slice empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl_any_conversions!(UintRef<'a>, 'a);

impl<'a> DecodeValue<'a> for UintRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        let bytes = BytesRef::decode_value(reader, header)?.as_slice();
        let result = Self::new(decode_to_slice(bytes)?)?;

        // Ensure we compute the same encoded length as the original any value.
        if result.value_len()? != header.length {
            return Err(Self::TAG.non_canonical_error());
        }

        Ok(result)
    }
}

impl<'a> EncodeValue for UintRef<'a> {
    fn value_len(&self) -> Result<Length> {
        encoded_len(self.inner.as_slice())
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        // Add leading `0x00` byte if required
        if self.value_len()? > self.len() {
            writer.write_byte(0)?;
        }

        writer.write(self.as_bytes())
    }
}

impl<'a> From<&UintRef<'a>> for UintRef<'a> {
    fn from(value: &UintRef<'a>) -> UintRef<'a> {
        *value
    }
}

impl<'a> FixedTag for UintRef<'a> {
    const TAG: Tag = Tag::Integer;
}

impl<'a> OrdIsValueOrd for UintRef<'a> {}

#[cfg(feature = "alloc")]
mod allocating {
    use super::{decode_to_slice, encoded_len, strip_leading_zeroes, UintRef};
    use crate::{
        ord::OrdIsValueOrd,
        referenced::{OwnedToRef, RefToOwned},
        BytesOwned, DecodeValue, EncodeValue, ErrorKind, FixedTag, Header, Length, Reader, Result,
        Tag, Writer,
    };

    /// Unsigned arbitrary precision ASN.1 `INTEGER` type.
    ///
    /// Provides heap-allocated storage for big endian bytes which comprise an
    /// unsigned integer value.
    ///
    /// Intended for use cases like very large integers that are used in
    /// cryptographic applications (e.g. keys, signatures).
    #[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
    pub struct Uint {
        /// Inner value
        inner: BytesOwned,
    }

    impl Uint {
        /// Create a new [`Uint`] from a byte slice.
        pub fn new(bytes: &[u8]) -> Result<Self> {
            let inner = BytesOwned::new(strip_leading_zeroes(bytes))
                .map_err(|_| ErrorKind::Length { tag: Self::TAG })?;

            Ok(Self { inner })
        }

        /// Borrow the inner byte slice which contains the least significant bytes
        /// of a big endian integer value with all leading zeros stripped.
        pub fn as_bytes(&self) -> &[u8] {
            self.inner.as_slice()
        }

        /// Get the length of this [`Uint`] in bytes.
        pub fn len(&self) -> Length {
            self.inner.len()
        }

        /// Is the inner byte slice empty?
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
    }

    impl_any_conversions!(Uint);

    impl<'a> DecodeValue<'a> for Uint {
        fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
            let bytes = BytesOwned::decode_value(reader, header)?;
            let result = Self::new(decode_to_slice(bytes.as_slice())?)?;

            // Ensure we compute the same encoded length as the original any value.
            if result.value_len()? != header.length {
                return Err(Self::TAG.non_canonical_error());
            }

            Ok(result)
        }
    }

    impl EncodeValue for Uint {
        fn value_len(&self) -> Result<Length> {
            encoded_len(self.inner.as_slice())
        }

        fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
            // Add leading `0x00` byte if required
            if self.value_len()? > self.len() {
                writer.write_byte(0)?;
            }

            writer.write(self.as_bytes())
        }
    }

    impl<'a> From<&UintRef<'a>> for Uint {
        fn from(value: &UintRef<'a>) -> Uint {
            let inner = BytesOwned::new(value.as_bytes()).expect("Invalid Uint");
            Uint { inner }
        }
    }

    impl FixedTag for Uint {
        const TAG: Tag = Tag::Integer;
    }

    impl OrdIsValueOrd for Uint {}

    impl<'a> RefToOwned<'a> for UintRef<'a> {
        type Owned = Uint;
        fn ref_to_owned(&self) -> Self::Owned {
            let inner = self.inner.ref_to_owned();

            Uint { inner }
        }
    }

    impl OwnedToRef for Uint {
        type Borrowed<'a> = UintRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            let inner = self.inner.owned_to_ref();

            UintRef { inner }
        }
    }
}

/// Decode an unsigned integer into a big endian byte slice with all leading
/// zeroes removed.
///
/// Returns a byte array of the requested size containing a big endian integer.
pub(crate) fn decode_to_slice(bytes: &[u8]) -> Result<&[u8]> {
    // The `INTEGER` type always encodes a signed value, so for unsigned
    // values the leading `0x00` byte may need to be removed.
    //
    // We also disallow a leading byte which would overflow a signed ASN.1
    // integer (since we're decoding an unsigned integer).
    // We expect all such cases to have a leading `0x00` byte.
    match bytes {
        [] => Err(Tag::Integer.non_canonical_error()),
        [0] => Ok(bytes),
        [0, byte, ..] if *byte < 0x80 => Err(Tag::Integer.non_canonical_error()),
        [0, rest @ ..] => Ok(rest),
        [byte, ..] if *byte >= 0x80 => Err(Tag::Integer.value_error()),
        _ => Ok(bytes),
    }
}

/// Decode an unsigned integer into a byte array of the requested size
/// containing a big endian integer.
pub(super) fn decode_to_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    let input = decode_to_slice(bytes)?;

    // Compute number of leading zeroes to add
    let num_zeroes = N
        .checked_sub(input.len())
        .ok_or_else(|| Tag::Integer.length_error())?;

    // Copy input into `N`-sized output buffer with leading zeroes
    let mut output = [0u8; N];
    output[num_zeroes..].copy_from_slice(input);
    Ok(output)
}

/// Encode the given big endian bytes representing an integer as ASN.1 DER.
pub(crate) fn encode_bytes<W>(encoder: &mut W, bytes: &[u8]) -> Result<()>
where
    W: Writer + ?Sized,
{
    let bytes = strip_leading_zeroes(bytes);

    if needs_leading_zero(bytes) {
        encoder.write_byte(0)?;
    }

    encoder.write(bytes)
}

/// Get the encoded length for the given unsigned integer serialized as bytes.
#[inline]
pub(crate) fn encoded_len(bytes: &[u8]) -> Result<Length> {
    let bytes = strip_leading_zeroes(bytes);
    Length::try_from(bytes.len())? + u8::from(needs_leading_zero(bytes))
}

/// Strip the leading zeroes from the given byte slice
pub(crate) fn strip_leading_zeroes(mut bytes: &[u8]) -> &[u8] {
    while let Some((byte, rest)) = bytes.split_first() {
        if *byte == 0 && !rest.is_empty() {
            bytes = rest;
        } else {
            break;
        }
    }

    bytes
}

/// Does the given integer need a leading zero?
fn needs_leading_zero(bytes: &[u8]) -> bool {
    matches!(bytes.first(), Some(byte) if *byte >= 0x80)
}

#[cfg(test)]
mod tests {
    use super::{decode_to_array, UintRef};
    use crate::{asn1::integer::tests::*, AnyRef, Decode, Encode, ErrorKind, SliceWriter, Tag};

    #[test]
    fn decode_to_array_no_leading_zero() {
        let arr = decode_to_array::<4>(&[1, 2]).unwrap();
        assert_eq!(arr, [0, 0, 1, 2]);
    }

    #[test]
    fn decode_to_array_leading_zero() {
        let arr = decode_to_array::<4>(&[0x00, 0xFF, 0xFE]).unwrap();
        assert_eq!(arr, [0x00, 0x00, 0xFF, 0xFE]);
    }

    #[test]
    fn decode_to_array_extra_zero() {
        let err = decode_to_array::<4>(&[0, 1, 2]).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::Noncanonical { tag: Tag::Integer });
    }

    #[test]
    fn decode_to_array_missing_zero() {
        // We're decoding an unsigned integer, but this value would be signed
        let err = decode_to_array::<4>(&[0xFF, 0xFE]).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::Value { tag: Tag::Integer });
    }

    #[test]
    fn decode_to_array_oversized_input() {
        let err = decode_to_array::<1>(&[1, 2, 3]).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::Length { tag: Tag::Integer });
    }

    #[test]
    fn decode_uintref() {
        assert_eq!(&[0], UintRef::from_der(I0_BYTES).unwrap().as_bytes());
        assert_eq!(&[127], UintRef::from_der(I127_BYTES).unwrap().as_bytes());
        assert_eq!(&[128], UintRef::from_der(I128_BYTES).unwrap().as_bytes());
        assert_eq!(&[255], UintRef::from_der(I255_BYTES).unwrap().as_bytes());

        assert_eq!(
            &[0x01, 0x00],
            UintRef::from_der(I256_BYTES).unwrap().as_bytes()
        );

        assert_eq!(
            &[0x7F, 0xFF],
            UintRef::from_der(I32767_BYTES).unwrap().as_bytes()
        );
    }

    #[test]
    fn encode_uintref() {
        for &example in &[
            I0_BYTES,
            I127_BYTES,
            I128_BYTES,
            I255_BYTES,
            I256_BYTES,
            I32767_BYTES,
        ] {
            let uint = UintRef::from_der(example).unwrap();

            let mut buf = [0u8; 128];
            let mut encoder = SliceWriter::new(&mut buf);
            uint.encode(&mut encoder).unwrap();

            let result = encoder.finish().unwrap();
            assert_eq!(example, result);
        }
    }

    #[test]
    fn reject_oversize_without_extra_zero() {
        let err = UintRef::try_from(AnyRef::new(Tag::Integer, &[0x81]).unwrap())
            .err()
            .unwrap();

        assert_eq!(err.kind(), ErrorKind::Value { tag: Tag::Integer });
    }
}
