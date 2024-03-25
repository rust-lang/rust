//! ASN.1 `OCTET STRING` support.

use crate::{
    asn1::AnyRef, ord::OrdIsValueOrd, BytesRef, Decode, DecodeValue, EncodeValue, ErrorKind,
    FixedTag, Header, Length, Reader, Result, Tag, Writer,
};

/// ASN.1 `OCTET STRING` type: borrowed form.
///
/// Octet strings represent contiguous sequences of octets, a.k.a. bytes.
///
/// This is a zero-copy reference type which borrows from the input data.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct OctetStringRef<'a> {
    /// Inner value
    inner: BytesRef<'a>,
}

impl<'a> OctetStringRef<'a> {
    /// Create a new ASN.1 `OCTET STRING` from a byte slice.
    pub fn new(slice: &'a [u8]) -> Result<Self> {
        BytesRef::new(slice)
            .map(|inner| Self { inner })
            .map_err(|_| ErrorKind::Length { tag: Self::TAG }.into())
    }

    /// Borrow the inner byte slice.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.inner.as_slice()
    }

    /// Get the length of the inner byte slice.
    pub fn len(&self) -> Length {
        self.inner.len()
    }

    /// Is the inner byte slice empty?
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Parse `T` from this `OCTET STRING`'s contents.
    pub fn decode_into<T: Decode<'a>>(&self) -> Result<T> {
        Decode::from_der(self.as_bytes())
    }
}

impl_any_conversions!(OctetStringRef<'a>, 'a);

impl AsRef<[u8]> for OctetStringRef<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> DecodeValue<'a> for OctetStringRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        let inner = BytesRef::decode_value(reader, header)?;
        Ok(Self { inner })
    }
}

impl EncodeValue for OctetStringRef<'_> {
    fn value_len(&self) -> Result<Length> {
        self.inner.value_len()
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        self.inner.encode_value(writer)
    }
}

impl FixedTag for OctetStringRef<'_> {
    const TAG: Tag = Tag::OctetString;
}

impl OrdIsValueOrd for OctetStringRef<'_> {}

impl<'a> From<&OctetStringRef<'a>> for OctetStringRef<'a> {
    fn from(value: &OctetStringRef<'a>) -> OctetStringRef<'a> {
        *value
    }
}

impl<'a> From<OctetStringRef<'a>> for AnyRef<'a> {
    fn from(octet_string: OctetStringRef<'a>) -> AnyRef<'a> {
        AnyRef::from_tag_and_value(Tag::OctetString, octet_string.inner)
    }
}

impl<'a> From<OctetStringRef<'a>> for &'a [u8] {
    fn from(octet_string: OctetStringRef<'a>) -> &'a [u8] {
        octet_string.as_bytes()
    }
}

#[cfg(feature = "alloc")]
pub use self::allocating::OctetString;

#[cfg(feature = "alloc")]
mod allocating {
    use super::*;
    use crate::referenced::*;
    use alloc::vec::Vec;

    /// ASN.1 `OCTET STRING` type: owned form..
    ///
    /// Octet strings represent contiguous sequences of octets, a.k.a. bytes.
    ///
    /// This type provides the same functionality as [`OctetStringRef`] but owns
    /// the backing data.
    #[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
    pub struct OctetString {
        /// Bitstring represented as a slice of bytes.
        pub(super) inner: Vec<u8>,
    }

    impl OctetString {
        /// Create a new ASN.1 `OCTET STRING`.
        pub fn new(bytes: impl Into<Vec<u8>>) -> Result<Self> {
            let inner = bytes.into();

            // Ensure the bytes parse successfully as an `OctetStringRef`
            OctetStringRef::new(&inner)?;

            Ok(Self { inner })
        }

        /// Borrow the inner byte slice.
        pub fn as_bytes(&self) -> &[u8] {
            self.inner.as_slice()
        }

        /// Take ownership of the octet string.
        pub fn into_bytes(self) -> Vec<u8> {
            self.inner
        }

        /// Get the length of the inner byte slice.
        pub fn len(&self) -> Length {
            self.value_len().expect("invalid OCTET STRING length")
        }

        /// Is the inner byte slice empty?
        pub fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }
    }

    impl_any_conversions!(OctetString);

    impl AsRef<[u8]> for OctetString {
        fn as_ref(&self) -> &[u8] {
            self.as_bytes()
        }
    }

    impl<'a> DecodeValue<'a> for OctetString {
        fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
            Self::new(reader.read_vec(header.length)?)
        }
    }

    impl EncodeValue for OctetString {
        fn value_len(&self) -> Result<Length> {
            self.inner.len().try_into()
        }

        fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
            writer.write(&self.inner)
        }
    }

    impl FixedTag for OctetString {
        const TAG: Tag = Tag::OctetString;
    }

    impl<'a> From<&'a OctetString> for OctetStringRef<'a> {
        fn from(octet_string: &'a OctetString) -> OctetStringRef<'a> {
            // Ensured to parse successfully in constructor
            OctetStringRef::new(&octet_string.inner).expect("invalid OCTET STRING")
        }
    }

    impl OrdIsValueOrd for OctetString {}

    impl<'a> RefToOwned<'a> for OctetStringRef<'a> {
        type Owned = OctetString;
        fn ref_to_owned(&self) -> Self::Owned {
            OctetString {
                inner: Vec::from(self.inner.as_slice()),
            }
        }
    }

    impl OwnedToRef for OctetString {
        type Borrowed<'a> = OctetStringRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            self.into()
        }
    }

    // Implement by hand because the derive would create invalid values.
    // Use the constructor to create a valid value.
    #[cfg(feature = "arbitrary")]
    impl<'a> arbitrary::Arbitrary<'a> for OctetString {
        fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
            Self::new(Vec::arbitrary(u)?).map_err(|_| arbitrary::Error::IncorrectFormat)
        }

        fn size_hint(depth: usize) -> (usize, Option<usize>) {
            arbitrary::size_hint::and(u8::size_hint(depth), Vec::<u8>::size_hint(depth))
        }
    }
}

#[cfg(feature = "bytes")]
mod bytes {
    use super::OctetString;
    use crate::{DecodeValue, EncodeValue, FixedTag, Header, Length, Reader, Result, Tag, Writer};
    use bytes::Bytes;

    impl<'a> DecodeValue<'a> for Bytes {
        fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
            OctetString::decode_value(reader, header).map(|octet_string| octet_string.inner.into())
        }
    }

    impl EncodeValue for Bytes {
        fn value_len(&self) -> Result<Length> {
            self.len().try_into()
        }

        fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
            writer.write(self.as_ref())
        }
    }

    impl FixedTag for Bytes {
        const TAG: Tag = Tag::OctetString;
    }
}

#[cfg(test)]
mod tests {
    use crate::asn1::{OctetStringRef, PrintableStringRef};

    #[test]
    fn octet_string_decode_into() {
        // PrintableString "hi"
        let der = b"\x13\x02\x68\x69";
        let oct = OctetStringRef::new(der).unwrap();

        let res = oct.decode_into::<PrintableStringRef<'_>>().unwrap();
        assert_eq!(AsRef::<str>::as_ref(&res), "hi");
    }
}
