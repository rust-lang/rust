//! Support for encoding signed integers

use super::{is_highest_bit_set, uint, value_cmp};
use crate::{
    ord::OrdIsValueOrd, AnyRef, BytesRef, DecodeValue, EncodeValue, Error, ErrorKind, FixedTag,
    Header, Length, Reader, Result, Tag, ValueOrd, Writer,
};
use core::cmp::Ordering;

#[cfg(feature = "alloc")]
pub use allocating::Int;

macro_rules! impl_encoding_traits {
    ($($int:ty => $uint:ty),+) => {
        $(
            impl<'a> DecodeValue<'a> for $int {
                fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
                    let mut buf = [0u8; Self::BITS as usize / 8];
                    let max_length = u32::from(header.length) as usize;

                    if max_length > buf.len() {
                        return Err(Self::TAG.non_canonical_error());
                    }

                    let bytes = reader.read_into(&mut buf[..max_length])?;

                    let result = if is_highest_bit_set(bytes) {
                        <$uint>::from_be_bytes(decode_to_array(bytes)?) as $int
                    } else {
                        Self::from_be_bytes(uint::decode_to_array(bytes)?)
                    };

                    // Ensure we compute the same encoded length as the original any value
                    if header.length != result.value_len()? {
                        return Err(Self::TAG.non_canonical_error());
                    }

                    Ok(result)
                }
            }

            impl EncodeValue for $int {
                fn value_len(&self) -> Result<Length> {
                    if *self < 0 {
                        negative_encoded_len(&(*self as $uint).to_be_bytes())
                    } else {
                        uint::encoded_len(&self.to_be_bytes())
                    }
                }

                fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
                    if *self < 0 {
                        encode_bytes(writer, &(*self as $uint).to_be_bytes())
                    } else {
                        uint::encode_bytes(writer, &self.to_be_bytes())
                    }
                }
            }

            impl FixedTag for $int {
                const TAG: Tag = Tag::Integer;
            }

            impl ValueOrd for $int {
                fn value_cmp(&self, other: &Self) -> Result<Ordering> {
                    value_cmp(*self, *other)
                }
            }

            impl TryFrom<AnyRef<'_>> for $int {
                type Error = Error;

                fn try_from(any: AnyRef<'_>) -> Result<Self> {
                    any.decode_as()
                }
            }
        )+
    };
}

impl_encoding_traits!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128);

/// Signed arbitrary precision ASN.1 `INTEGER` reference type.
///
/// Provides direct access to the underlying big endian bytes which comprise
/// an signed integer value.
///
/// Intended for use cases like very large integers that are used in
/// cryptographic applications (e.g. keys, signatures).
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct IntRef<'a> {
    /// Inner value
    inner: BytesRef<'a>,
}

impl<'a> IntRef<'a> {
    /// Create a new [`IntRef`] from a byte slice.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        let inner = BytesRef::new(strip_leading_ones(bytes))
            .map_err(|_| ErrorKind::Length { tag: Self::TAG })?;

        Ok(Self { inner })
    }

    /// Borrow the inner byte slice which contains the least significant bytes
    /// of a big endian integer value with all leading ones stripped.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.inner.as_slice()
    }

    /// Get the length of this [`IntRef`] in bytes.
    pub fn len(&self) -> Length {
        self.inner.len()
    }

    /// Is the inner byte slice empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl_any_conversions!(IntRef<'a>, 'a);

impl<'a> DecodeValue<'a> for IntRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        let bytes = BytesRef::decode_value(reader, header)?;
        validate_canonical(bytes.as_slice())?;

        let result = Self::new(bytes.as_slice())?;

        // Ensure we compute the same encoded length as the original any value.
        if result.value_len()? != header.length {
            return Err(Self::TAG.non_canonical_error());
        }

        Ok(result)
    }
}

impl<'a> EncodeValue for IntRef<'a> {
    fn value_len(&self) -> Result<Length> {
        // Signed integers always hold their full encoded form.
        Ok(self.inner.len())
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.as_bytes())
    }
}

impl<'a> From<&IntRef<'a>> for IntRef<'a> {
    fn from(value: &IntRef<'a>) -> IntRef<'a> {
        *value
    }
}

impl<'a> FixedTag for IntRef<'a> {
    const TAG: Tag = Tag::Integer;
}

impl<'a> OrdIsValueOrd for IntRef<'a> {}

#[cfg(feature = "alloc")]
mod allocating {
    use super::{strip_leading_ones, validate_canonical, IntRef};
    use crate::{
        asn1::Uint,
        ord::OrdIsValueOrd,
        referenced::{OwnedToRef, RefToOwned},
        BytesOwned, DecodeValue, EncodeValue, ErrorKind, FixedTag, Header, Length, Reader, Result,
        Tag, Writer,
    };
    use alloc::vec::Vec;

    /// Signed arbitrary precision ASN.1 `INTEGER` type.
    ///
    /// Provides heap-allocated storage for big endian bytes which comprise an
    /// signed integer value.
    ///
    /// Intended for use cases like very large integers that are used in
    /// cryptographic applications (e.g. keys, signatures).
    #[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
    pub struct Int {
        /// Inner value
        inner: BytesOwned,
    }

    impl Int {
        /// Create a new [`Int`] from a byte slice.
        pub fn new(bytes: &[u8]) -> Result<Self> {
            let inner = BytesOwned::new(strip_leading_ones(bytes))
                .map_err(|_| ErrorKind::Length { tag: Self::TAG })?;

            Ok(Self { inner })
        }

        /// Borrow the inner byte slice which contains the least significant bytes
        /// of a big endian integer value with all leading ones stripped.
        pub fn as_bytes(&self) -> &[u8] {
            self.inner.as_slice()
        }

        /// Get the length of this [`Int`] in bytes.
        pub fn len(&self) -> Length {
            self.inner.len()
        }

        /// Is the inner byte slice empty?
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
    }

    impl_any_conversions!(Int);

    impl<'a> DecodeValue<'a> for Int {
        fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
            let bytes = BytesOwned::decode_value(reader, header)?;
            validate_canonical(bytes.as_slice())?;

            let result = Self::new(bytes.as_slice())?;

            // Ensure we compute the same encoded length as the original any value.
            if result.value_len()? != header.length {
                return Err(Self::TAG.non_canonical_error());
            }

            Ok(result)
        }
    }

    impl EncodeValue for Int {
        fn value_len(&self) -> Result<Length> {
            // Signed integers always hold their full encoded form.
            Ok(self.inner.len())
        }

        fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
            writer.write(self.as_bytes())
        }
    }

    impl<'a> From<&IntRef<'a>> for Int {
        fn from(value: &IntRef<'a>) -> Int {
            let inner = BytesOwned::new(value.as_bytes()).expect("Invalid Int");
            Int { inner }
        }
    }

    impl From<Uint> for Int {
        fn from(value: Uint) -> Self {
            let mut inner: Vec<u8> = Vec::new();

            // Add leading `0x00` byte if required
            if value.value_len().expect("invalid Uint") > value.len() {
                inner.push(0x00);
            }

            inner.extend_from_slice(value.as_bytes());
            let inner = BytesOwned::new(inner).expect("invalid Uint");

            Int { inner }
        }
    }

    impl FixedTag for Int {
        const TAG: Tag = Tag::Integer;
    }

    impl OrdIsValueOrd for Int {}

    impl<'a> RefToOwned<'a> for IntRef<'a> {
        type Owned = Int;
        fn ref_to_owned(&self) -> Self::Owned {
            let inner = self.inner.ref_to_owned();

            Int { inner }
        }
    }

    impl OwnedToRef for Int {
        type Borrowed<'a> = IntRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            let inner = self.inner.owned_to_ref();

            IntRef { inner }
        }
    }
}

/// Ensure `INTEGER` is canonically encoded.
fn validate_canonical(bytes: &[u8]) -> Result<()> {
    // The `INTEGER` type always encodes a signed value and we're decoding
    // as signed here, so we allow a zero extension or sign extension byte,
    // but only as permitted under DER canonicalization.
    match bytes {
        [] => Err(Tag::Integer.non_canonical_error()),
        [0x00, byte, ..] if *byte < 0x80 => Err(Tag::Integer.non_canonical_error()),
        [0xFF, byte, ..] if *byte >= 0x80 => Err(Tag::Integer.non_canonical_error()),
        _ => Ok(()),
    }
}

/// Decode an signed integer of the specified size.
///
/// Returns a byte array of the requested size containing a big endian integer.
fn decode_to_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    match N.checked_sub(bytes.len()) {
        Some(offset) => {
            let mut output = [0xFFu8; N];
            output[offset..].copy_from_slice(bytes);
            Ok(output)
        }
        None => {
            let expected_len = Length::try_from(N)?;
            let actual_len = Length::try_from(bytes.len())?;

            Err(ErrorKind::Incomplete {
                expected_len,
                actual_len,
            }
            .into())
        }
    }
}

/// Encode the given big endian bytes representing an integer as ASN.1 DER.
fn encode_bytes<W>(writer: &mut W, bytes: &[u8]) -> Result<()>
where
    W: Writer + ?Sized,
{
    writer.write(strip_leading_ones(bytes))
}

/// Get the encoded length for the given **negative** integer serialized as bytes.
#[inline]
fn negative_encoded_len(bytes: &[u8]) -> Result<Length> {
    Length::try_from(strip_leading_ones(bytes).len())
}

/// Strip the leading all-ones bytes from the given byte slice.
pub(crate) fn strip_leading_ones(mut bytes: &[u8]) -> &[u8] {
    while let Some((byte, rest)) = bytes.split_first() {
        if *byte == 0xFF && is_highest_bit_set(rest) {
            bytes = rest;
            continue;
        }

        break;
    }

    bytes
}

#[cfg(test)]
mod tests {
    use super::{validate_canonical, IntRef};
    use crate::{asn1::integer::tests::*, Decode, Encode, SliceWriter};

    #[test]
    fn validate_canonical_ok() {
        assert_eq!(validate_canonical(&[0x00]), Ok(()));
        assert_eq!(validate_canonical(&[0x01]), Ok(()));
        assert_eq!(validate_canonical(&[0x00, 0x80]), Ok(()));
        assert_eq!(validate_canonical(&[0xFF, 0x00]), Ok(()));
    }

    #[test]
    fn validate_canonical_err() {
        // Empty integers are always non-canonical.
        assert!(validate_canonical(&[]).is_err());

        // Positives with excessive zero extension are non-canonical.
        assert!(validate_canonical(&[0x00, 0x00]).is_err());

        // Negatives with excessive sign extension are non-canonical.
        assert!(validate_canonical(&[0xFF, 0x80]).is_err());
    }

    #[test]
    fn decode_intref() {
        // Positive numbers decode, but have zero extensions as necessary
        // (to distinguish them from negative representations).
        assert_eq!(&[0], IntRef::from_der(I0_BYTES).unwrap().as_bytes());
        assert_eq!(&[127], IntRef::from_der(I127_BYTES).unwrap().as_bytes());
        assert_eq!(&[0, 128], IntRef::from_der(I128_BYTES).unwrap().as_bytes());
        assert_eq!(&[0, 255], IntRef::from_der(I255_BYTES).unwrap().as_bytes());

        assert_eq!(
            &[0x01, 0x00],
            IntRef::from_der(I256_BYTES).unwrap().as_bytes()
        );

        assert_eq!(
            &[0x7F, 0xFF],
            IntRef::from_der(I32767_BYTES).unwrap().as_bytes()
        );

        // Negative integers decode.
        assert_eq!(&[128], IntRef::from_der(INEG128_BYTES).unwrap().as_bytes());
        assert_eq!(
            &[255, 127],
            IntRef::from_der(INEG129_BYTES).unwrap().as_bytes()
        );
        assert_eq!(
            &[128, 0],
            IntRef::from_der(INEG32768_BYTES).unwrap().as_bytes()
        );
    }

    #[test]
    fn encode_intref() {
        for &example in &[
            I0_BYTES,
            I127_BYTES,
            I128_BYTES,
            I255_BYTES,
            I256_BYTES,
            I32767_BYTES,
        ] {
            let uint = IntRef::from_der(example).unwrap();

            let mut buf = [0u8; 128];
            let mut encoder = SliceWriter::new(&mut buf);
            uint.encode(&mut encoder).unwrap();

            let result = encoder.finish().unwrap();
            assert_eq!(example, result);
        }

        for &example in &[INEG128_BYTES, INEG129_BYTES, INEG32768_BYTES] {
            let uint = IntRef::from_der(example).unwrap();

            let mut buf = [0u8; 128];
            let mut encoder = SliceWriter::new(&mut buf);
            uint.encode(&mut encoder).unwrap();

            let result = encoder.finish().unwrap();
            assert_eq!(example, result);
        }
    }
}
