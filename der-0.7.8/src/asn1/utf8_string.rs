//! ASN.1 `UTF8String` support.

use crate::{
    asn1::AnyRef, ord::OrdIsValueOrd, EncodeValue, Error, FixedTag, Length, Result, StrRef, Tag,
    Writer,
};
use core::{fmt, ops::Deref, str};

#[cfg(feature = "alloc")]
use {
    crate::{DecodeValue, Header, Reader},
    alloc::{borrow::ToOwned, string::String},
};

/// ASN.1 `UTF8String` type.
///
/// Supports the full UTF-8 encoding.
///
/// Note that the [`Decode`][`crate::Decode`] and [`Encode`][`crate::Encode`]
/// traits are impl'd for Rust's [`str`][`prim@str`] primitive, which
/// decodes/encodes as a [`Utf8StringRef`].
///
/// You are free to use [`str`][`prim@str`] instead of this type, however it's
/// still provided for explicitness in cases where it might be ambiguous with
/// other ASN.1 string encodings such as
/// [`PrintableStringRef`][`crate::asn1::PrintableStringRef`].
///
/// This is a zero-copy reference type which borrows from the input data.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Utf8StringRef<'a> {
    /// Inner value
    inner: StrRef<'a>,
}

impl<'a> Utf8StringRef<'a> {
    /// Create a new ASN.1 `UTF8String`.
    pub fn new<T>(input: &'a T) -> Result<Self>
    where
        T: AsRef<[u8]> + ?Sized,
    {
        StrRef::from_bytes(input.as_ref()).map(|inner| Self { inner })
    }
}

impl_string_type!(Utf8StringRef<'a>, 'a);

impl<'a> Deref for Utf8StringRef<'a> {
    type Target = StrRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl FixedTag for Utf8StringRef<'_> {
    const TAG: Tag = Tag::Utf8String;
}

impl<'a> From<&Utf8StringRef<'a>> for Utf8StringRef<'a> {
    fn from(value: &Utf8StringRef<'a>) -> Utf8StringRef<'a> {
        *value
    }
}

impl<'a> From<Utf8StringRef<'a>> for AnyRef<'a> {
    fn from(utf_string: Utf8StringRef<'a>) -> AnyRef<'a> {
        AnyRef::from_tag_and_value(Tag::Utf8String, utf_string.inner.into())
    }
}

impl<'a> fmt::Debug for Utf8StringRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Utf8String({:?})", self.as_str())
    }
}

impl<'a> TryFrom<AnyRef<'a>> for &'a str {
    type Error = Error;

    fn try_from(any: AnyRef<'a>) -> Result<&'a str> {
        Utf8StringRef::try_from(any).map(|s| s.as_str())
    }
}

impl EncodeValue for str {
    fn value_len(&self) -> Result<Length> {
        Utf8StringRef::new(self)?.value_len()
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        Utf8StringRef::new(self)?.encode_value(writer)
    }
}

impl FixedTag for str {
    const TAG: Tag = Tag::Utf8String;
}

impl OrdIsValueOrd for str {}

#[cfg(feature = "alloc")]
impl<'a> From<Utf8StringRef<'a>> for String {
    fn from(s: Utf8StringRef<'a>) -> String {
        s.as_str().to_owned()
    }
}

#[cfg(feature = "alloc")]
impl<'a> TryFrom<AnyRef<'a>> for String {
    type Error = Error;

    fn try_from(any: AnyRef<'a>) -> Result<String> {
        Utf8StringRef::try_from(any).map(|s| s.as_str().to_owned())
    }
}

#[cfg(feature = "alloc")]
impl<'a> DecodeValue<'a> for String {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        Ok(String::from_utf8(reader.read_vec(header.length)?)?)
    }
}

#[cfg(feature = "alloc")]
impl EncodeValue for String {
    fn value_len(&self) -> Result<Length> {
        Utf8StringRef::new(self)?.value_len()
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        Utf8StringRef::new(self)?.encode_value(writer)
    }
}

#[cfg(feature = "alloc")]
impl FixedTag for String {
    const TAG: Tag = Tag::Utf8String;
}

#[cfg(feature = "alloc")]
impl OrdIsValueOrd for String {}

#[cfg(test)]
mod tests {
    use super::Utf8StringRef;
    use crate::Decode;

    #[test]
    fn parse_ascii_bytes() {
        let example_bytes = &[
            0x0c, 0x0b, 0x54, 0x65, 0x73, 0x74, 0x20, 0x55, 0x73, 0x65, 0x72, 0x20, 0x31,
        ];

        let utf8_string = Utf8StringRef::from_der(example_bytes).unwrap();
        assert_eq!(utf8_string.as_str(), "Test User 1");
    }

    #[test]
    fn parse_utf8_bytes() {
        let example_bytes = &[0x0c, 0x06, 0x48, 0x65, 0x6c, 0x6c, 0xc3, 0xb3];
        let utf8_string = Utf8StringRef::from_der(example_bytes).unwrap();
        assert_eq!(utf8_string.as_str(), "Helló");
    }
}
