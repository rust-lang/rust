//! ASN.1 `ANY` type.

#![cfg_attr(feature = "arbitrary", allow(clippy::integer_arithmetic))]

use crate::{
    BytesRef, Choice, Decode, DecodeValue, DerOrd, EncodeValue, Error, ErrorKind, Header, Length,
    Reader, Result, SliceReader, Tag, Tagged, ValueOrd, Writer,
};
use core::cmp::Ordering;

#[cfg(feature = "alloc")]
use crate::SliceWriter;

/// ASN.1 `ANY`: represents any explicitly tagged ASN.1 value.
///
/// This is a zero-copy reference type which borrows from the input data.
///
/// Technically `ANY` hasn't been a recommended part of ASN.1 since the X.209
/// revision from 1988. It was deprecated and replaced by Information Object
/// Classes in X.680 in 1994, and X.690 no longer refers to it whatsoever.
///
/// Nevertheless, this crate defines an `ANY` type as it remains a familiar
/// and useful concept which is still extensively used in things like
/// PKI-related RFCs.
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct AnyRef<'a> {
    /// Tag representing the type of the encoded value.
    tag: Tag,

    /// Inner value encoded as bytes.
    value: BytesRef<'a>,
}

impl<'a> AnyRef<'a> {
    /// [`AnyRef`] representation of the ASN.1 `NULL` type.
    pub const NULL: Self = Self {
        tag: Tag::Null,
        value: BytesRef::EMPTY,
    };

    /// Create a new [`AnyRef`] from the provided [`Tag`] and DER bytes.
    pub fn new(tag: Tag, bytes: &'a [u8]) -> Result<Self> {
        let value = BytesRef::new(bytes).map_err(|_| ErrorKind::Length { tag })?;
        Ok(Self { tag, value })
    }

    /// Infallible creation of an [`AnyRef`] from a [`BytesRef`].
    pub(crate) fn from_tag_and_value(tag: Tag, value: BytesRef<'a>) -> Self {
        Self { tag, value }
    }

    /// Get the raw value for this [`AnyRef`] type as a byte slice.
    pub fn value(self) -> &'a [u8] {
        self.value.as_slice()
    }

    /// Attempt to decode this [`AnyRef`] type into the inner value.
    pub fn decode_as<T>(self) -> Result<T>
    where
        T: Choice<'a> + DecodeValue<'a>,
    {
        if !T::can_decode(self.tag) {
            return Err(self.tag.unexpected_error(None));
        }

        let header = Header {
            tag: self.tag,
            length: self.value.len(),
        };

        let mut decoder = SliceReader::new(self.value())?;
        let result = T::decode_value(&mut decoder, header)?;
        decoder.finish(result)
    }

    /// Is this value an ASN.1 `NULL` value?
    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    /// Attempt to decode this value an ASN.1 `SEQUENCE`, creating a new
    /// nested reader and calling the provided argument with it.
    pub fn sequence<F, T>(self, f: F) -> Result<T>
    where
        F: FnOnce(&mut SliceReader<'a>) -> Result<T>,
    {
        self.tag.assert_eq(Tag::Sequence)?;
        let mut reader = SliceReader::new(self.value.as_slice())?;
        let result = f(&mut reader)?;
        reader.finish(result)
    }
}

impl<'a> Choice<'a> for AnyRef<'a> {
    fn can_decode(_: Tag) -> bool {
        true
    }
}

impl<'a> Decode<'a> for AnyRef<'a> {
    fn decode<R: Reader<'a>>(reader: &mut R) -> Result<AnyRef<'a>> {
        let header = Header::decode(reader)?;
        Self::decode_value(reader, header)
    }
}

impl<'a> DecodeValue<'a> for AnyRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        Ok(Self {
            tag: header.tag,
            value: BytesRef::decode_value(reader, header)?,
        })
    }
}

impl EncodeValue for AnyRef<'_> {
    fn value_len(&self) -> Result<Length> {
        Ok(self.value.len())
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.value())
    }
}

impl Tagged for AnyRef<'_> {
    fn tag(&self) -> Tag {
        self.tag
    }
}

impl ValueOrd for AnyRef<'_> {
    fn value_cmp(&self, other: &Self) -> Result<Ordering> {
        self.value.der_cmp(&other.value)
    }
}

impl<'a> From<AnyRef<'a>> for BytesRef<'a> {
    fn from(any: AnyRef<'a>) -> BytesRef<'a> {
        any.value
    }
}

impl<'a> TryFrom<&'a [u8]> for AnyRef<'a> {
    type Error = Error;

    fn try_from(bytes: &'a [u8]) -> Result<AnyRef<'a>> {
        AnyRef::from_der(bytes)
    }
}

#[cfg(feature = "alloc")]
pub use self::allocating::Any;

#[cfg(feature = "alloc")]
mod allocating {
    use super::*;
    use crate::{referenced::*, BytesOwned};
    use alloc::boxed::Box;

    /// ASN.1 `ANY`: represents any explicitly tagged ASN.1 value.
    ///
    /// This type provides the same functionality as [`AnyRef`] but owns the
    /// backing data.
    #[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
    #[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
    pub struct Any {
        /// Tag representing the type of the encoded value.
        tag: Tag,

        /// Inner value encoded as bytes.
        value: BytesOwned,
    }

    impl Any {
        /// Create a new [`Any`] from the provided [`Tag`] and DER bytes.
        pub fn new(tag: Tag, bytes: impl Into<Box<[u8]>>) -> Result<Self> {
            let value = BytesOwned::new(bytes)?;

            // Ensure the tag and value are a valid `AnyRef`.
            AnyRef::new(tag, value.as_slice())?;
            Ok(Self { tag, value })
        }

        /// Allow access to value
        pub fn value(&self) -> &[u8] {
            self.value.as_slice()
        }

        /// Attempt to decode this [`Any`] type into the inner value.
        pub fn decode_as<'a, T>(&'a self) -> Result<T>
        where
            T: Choice<'a> + DecodeValue<'a>,
        {
            AnyRef::from(self).decode_as()
        }

        /// Encode the provided type as an [`Any`] value.
        pub fn encode_from<T>(msg: &T) -> Result<Self>
        where
            T: Tagged + EncodeValue,
        {
            let encoded_len = usize::try_from(msg.value_len()?)?;
            let mut buf = vec![0u8; encoded_len];
            let mut writer = SliceWriter::new(&mut buf);
            msg.encode_value(&mut writer)?;
            writer.finish()?;
            Any::new(msg.tag(), buf)
        }

        /// Attempt to decode this value an ASN.1 `SEQUENCE`, creating a new
        /// nested reader and calling the provided argument with it.
        pub fn sequence<'a, F, T>(&'a self, f: F) -> Result<T>
        where
            F: FnOnce(&mut SliceReader<'a>) -> Result<T>,
        {
            AnyRef::from(self).sequence(f)
        }

        /// [`Any`] representation of the ASN.1 `NULL` type.
        pub fn null() -> Self {
            Self {
                tag: Tag::Null,
                value: BytesOwned::default(),
            }
        }
    }

    impl Choice<'_> for Any {
        fn can_decode(_: Tag) -> bool {
            true
        }
    }

    impl<'a> Decode<'a> for Any {
        fn decode<R: Reader<'a>>(reader: &mut R) -> Result<Self> {
            let header = Header::decode(reader)?;
            Self::decode_value(reader, header)
        }
    }

    impl<'a> DecodeValue<'a> for Any {
        fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
            let value = reader.read_vec(header.length)?;
            Self::new(header.tag, value)
        }
    }

    impl EncodeValue for Any {
        fn value_len(&self) -> Result<Length> {
            Ok(self.value.len())
        }

        fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
            writer.write(self.value.as_slice())
        }
    }

    impl<'a> From<&'a Any> for AnyRef<'a> {
        fn from(any: &'a Any) -> AnyRef<'a> {
            // Ensured to parse successfully in constructor
            AnyRef::new(any.tag, any.value.as_slice()).expect("invalid ANY")
        }
    }

    impl Tagged for Any {
        fn tag(&self) -> Tag {
            self.tag
        }
    }

    impl ValueOrd for Any {
        fn value_cmp(&self, other: &Self) -> Result<Ordering> {
            self.value.der_cmp(&other.value)
        }
    }

    impl<'a, T> From<T> for Any
    where
        T: Into<AnyRef<'a>>,
    {
        fn from(input: T) -> Any {
            let anyref: AnyRef<'a> = input.into();
            Self {
                tag: anyref.tag(),
                value: BytesOwned::from(anyref.value),
            }
        }
    }

    impl<'a> RefToOwned<'a> for AnyRef<'a> {
        type Owned = Any;
        fn ref_to_owned(&self) -> Self::Owned {
            Any {
                tag: self.tag(),
                value: BytesOwned::from(self.value),
            }
        }
    }

    impl OwnedToRef for Any {
        type Borrowed<'a> = AnyRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            self.into()
        }
    }

    impl Any {
        /// Is this value an ASN.1 `NULL` value?
        pub fn is_null(&self) -> bool {
            self.owned_to_ref() == AnyRef::NULL
        }
    }
}
