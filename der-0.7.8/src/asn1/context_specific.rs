//! Context-specific field.

use crate::{
    asn1::AnyRef, Choice, Decode, DecodeValue, DerOrd, Encode, EncodeValue, EncodeValueRef, Error,
    Header, Length, Reader, Result, Tag, TagMode, TagNumber, Tagged, ValueOrd, Writer,
};
use core::cmp::Ordering;

/// Context-specific field which wraps an owned inner value.
///
/// This type decodes/encodes a field which is specific to a particular context
/// and is identified by a [`TagNumber`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct ContextSpecific<T> {
    /// Context-specific tag number sans the leading `0b10000000` class
    /// identifier bit and `0b100000` constructed flag.
    pub tag_number: TagNumber,

    /// Tag mode: `EXPLICIT` VS `IMPLICIT`.
    pub tag_mode: TagMode,

    /// Value of the field.
    pub value: T,
}

impl<T> ContextSpecific<T> {
    /// Attempt to decode an `EXPLICIT` ASN.1 `CONTEXT-SPECIFIC` field with the
    /// provided [`TagNumber`].
    ///
    /// This method has the following behavior which is designed to simplify
    /// handling of extension fields, which are denoted in an ASN.1 schema
    /// using the `...` ellipsis extension marker:
    ///
    /// - Skips over [`ContextSpecific`] fields with a tag number lower than
    ///   the current one, consuming and ignoring them.
    /// - Returns `Ok(None)` if a [`ContextSpecific`] field with a higher tag
    ///   number is encountered. These fields are not consumed in this case,
    ///   allowing a field with a lower tag number to be omitted, then the
    ///   higher numbered field consumed as a follow-up.
    /// - Returns `Ok(None)` if anything other than a [`ContextSpecific`] field
    ///   is encountered.
    pub fn decode_explicit<'a, R: Reader<'a>>(
        reader: &mut R,
        tag_number: TagNumber,
    ) -> Result<Option<Self>>
    where
        T: Decode<'a>,
    {
        Self::decode_with(reader, tag_number, |reader| Self::decode(reader))
    }

    /// Attempt to decode an `IMPLICIT` ASN.1 `CONTEXT-SPECIFIC` field with the
    /// provided [`TagNumber`].
    ///
    /// This method otherwise behaves the same as `decode_explicit`,
    /// but should be used in cases where the particular fields are `IMPLICIT`
    /// as opposed to `EXPLICIT`.
    pub fn decode_implicit<'a, R: Reader<'a>>(
        reader: &mut R,
        tag_number: TagNumber,
    ) -> Result<Option<Self>>
    where
        T: DecodeValue<'a> + Tagged,
    {
        Self::decode_with(reader, tag_number, |reader| {
            let header = Header::decode(reader)?;
            let value = T::decode_value(reader, header)?;

            if header.tag.is_constructed() != value.tag().is_constructed() {
                return Err(header.tag.non_canonical_error());
            }

            Ok(Self {
                tag_number,
                tag_mode: TagMode::Implicit,
                value,
            })
        })
    }

    /// Attempt to decode a context-specific field with the given
    /// helper callback.
    fn decode_with<'a, F, R: Reader<'a>>(
        reader: &mut R,
        tag_number: TagNumber,
        f: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&mut R) -> Result<Self>,
    {
        while let Some(octet) = reader.peek_byte() {
            let tag = Tag::try_from(octet)?;

            if !tag.is_context_specific() || (tag.number() > tag_number) {
                break;
            } else if tag.number() == tag_number {
                return Some(f(reader)).transpose();
            } else {
                AnyRef::decode(reader)?;
            }
        }

        Ok(None)
    }
}

impl<'a, T> Choice<'a> for ContextSpecific<T>
where
    T: Decode<'a> + Tagged,
{
    fn can_decode(tag: Tag) -> bool {
        tag.is_context_specific()
    }
}

impl<'a, T> Decode<'a> for ContextSpecific<T>
where
    T: Decode<'a>,
{
    fn decode<R: Reader<'a>>(reader: &mut R) -> Result<Self> {
        let header = Header::decode(reader)?;

        match header.tag {
            Tag::ContextSpecific {
                number,
                constructed: true,
            } => Ok(Self {
                tag_number: number,
                tag_mode: TagMode::default(),
                value: reader.read_nested(header.length, |reader| T::decode(reader))?,
            }),
            tag => Err(tag.unexpected_error(None)),
        }
    }
}

impl<T> EncodeValue for ContextSpecific<T>
where
    T: EncodeValue + Tagged,
{
    fn value_len(&self) -> Result<Length> {
        match self.tag_mode {
            TagMode::Explicit => self.value.encoded_len(),
            TagMode::Implicit => self.value.value_len(),
        }
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        match self.tag_mode {
            TagMode::Explicit => self.value.encode(writer),
            TagMode::Implicit => self.value.encode_value(writer),
        }
    }
}

impl<T> Tagged for ContextSpecific<T>
where
    T: Tagged,
{
    fn tag(&self) -> Tag {
        let constructed = match self.tag_mode {
            TagMode::Explicit => true,
            TagMode::Implicit => self.value.tag().is_constructed(),
        };

        Tag::ContextSpecific {
            number: self.tag_number,
            constructed,
        }
    }
}

impl<'a, T> TryFrom<AnyRef<'a>> for ContextSpecific<T>
where
    T: Decode<'a>,
{
    type Error = Error;

    fn try_from(any: AnyRef<'a>) -> Result<ContextSpecific<T>> {
        match any.tag() {
            Tag::ContextSpecific {
                number,
                constructed: true,
            } => Ok(Self {
                tag_number: number,
                tag_mode: TagMode::default(),
                value: T::from_der(any.value())?,
            }),
            tag => Err(tag.unexpected_error(None)),
        }
    }
}

impl<T> ValueOrd for ContextSpecific<T>
where
    T: EncodeValue + ValueOrd + Tagged,
{
    fn value_cmp(&self, other: &Self) -> Result<Ordering> {
        match self.tag_mode {
            TagMode::Explicit => self.der_cmp(other),
            TagMode::Implicit => self.value_cmp(other),
        }
    }
}

/// Context-specific field reference.
///
/// This type encodes a field which is specific to a particular context
/// and is identified by a [`TagNumber`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct ContextSpecificRef<'a, T> {
    /// Context-specific tag number sans the leading `0b10000000` class
    /// identifier bit and `0b100000` constructed flag.
    pub tag_number: TagNumber,

    /// Tag mode: `EXPLICIT` VS `IMPLICIT`.
    pub tag_mode: TagMode,

    /// Value of the field.
    pub value: &'a T,
}

impl<'a, T> ContextSpecificRef<'a, T> {
    /// Convert to a [`ContextSpecific`].
    fn encoder(&self) -> ContextSpecific<EncodeValueRef<'a, T>> {
        ContextSpecific {
            tag_number: self.tag_number,
            tag_mode: self.tag_mode,
            value: EncodeValueRef(self.value),
        }
    }
}

impl<'a, T> EncodeValue for ContextSpecificRef<'a, T>
where
    T: EncodeValue + Tagged,
{
    fn value_len(&self) -> Result<Length> {
        self.encoder().value_len()
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        self.encoder().encode_value(writer)
    }
}

impl<'a, T> Tagged for ContextSpecificRef<'a, T>
where
    T: Tagged,
{
    fn tag(&self) -> Tag {
        self.encoder().tag()
    }
}

#[cfg(test)]
mod tests {
    use super::ContextSpecific;
    use crate::{asn1::BitStringRef, Decode, Encode, SliceReader, TagMode, TagNumber};
    use hex_literal::hex;

    // Public key data from `pkcs8` crate's `ed25519-pkcs8-v2.der`
    const EXAMPLE_BYTES: &[u8] =
        &hex!("A123032100A3A7EAE3A8373830BC47E1167BC50E1DB551999651E0E2DC587623438EAC3F31");

    #[test]
    fn round_trip() {
        let field = ContextSpecific::<BitStringRef<'_>>::from_der(EXAMPLE_BYTES).unwrap();
        assert_eq!(field.tag_number.value(), 1);
        assert_eq!(
            field.value,
            BitStringRef::from_bytes(&EXAMPLE_BYTES[5..]).unwrap()
        );

        let mut buf = [0u8; 128];
        let encoded = field.encode_to_slice(&mut buf).unwrap();
        assert_eq!(encoded, EXAMPLE_BYTES);
    }

    #[test]
    fn context_specific_with_explicit_field() {
        let tag_number = TagNumber::new(0);

        // Empty message
        let mut reader = SliceReader::new(&[]).unwrap();
        assert_eq!(
            ContextSpecific::<u8>::decode_explicit(&mut reader, tag_number).unwrap(),
            None
        );

        // Message containing a non-context-specific type
        let mut reader = SliceReader::new(&hex!("020100")).unwrap();
        assert_eq!(
            ContextSpecific::<u8>::decode_explicit(&mut reader, tag_number).unwrap(),
            None
        );

        // Message containing an EXPLICIT context-specific field
        let mut reader = SliceReader::new(&hex!("A003020100")).unwrap();
        let field = ContextSpecific::<u8>::decode_explicit(&mut reader, tag_number)
            .unwrap()
            .unwrap();

        assert_eq!(field.tag_number, tag_number);
        assert_eq!(field.tag_mode, TagMode::Explicit);
        assert_eq!(field.value, 0);
    }

    #[test]
    fn context_specific_with_implicit_field() {
        // From RFC8410 Section 10.3:
        // <https://datatracker.ietf.org/doc/html/rfc8410#section-10.3>
        //
        //    81  33:   [1] 00 19 BF 44 09 69 84 CD FE 85 41 BA C1 67 DC 3B
        //                  96 C8 50 86 AA 30 B6 B6 CB 0C 5C 38 AD 70 31 66
        //                  E1
        let context_specific_implicit_bytes =
            hex!("81210019BF44096984CDFE8541BAC167DC3B96C85086AA30B6B6CB0C5C38AD703166E1");

        let tag_number = TagNumber::new(1);

        let mut reader = SliceReader::new(&context_specific_implicit_bytes).unwrap();
        let field = ContextSpecific::<BitStringRef<'_>>::decode_implicit(&mut reader, tag_number)
            .unwrap()
            .unwrap();

        assert_eq!(field.tag_number, tag_number);
        assert_eq!(field.tag_mode, TagMode::Implicit);
        assert_eq!(
            field.value.as_bytes().unwrap(),
            &context_specific_implicit_bytes[3..]
        );
    }

    #[test]
    fn context_specific_skipping_unknown_field() {
        let tag = TagNumber::new(1);
        let mut reader = SliceReader::new(&hex!("A003020100A103020101")).unwrap();
        let field = ContextSpecific::<u8>::decode_explicit(&mut reader, tag)
            .unwrap()
            .unwrap();
        assert_eq!(field.value, 1);
    }

    #[test]
    fn context_specific_returns_none_on_greater_tag_number() {
        let tag = TagNumber::new(0);
        let mut reader = SliceReader::new(&hex!("A103020101")).unwrap();
        assert_eq!(
            ContextSpecific::<u8>::decode_explicit(&mut reader, tag).unwrap(),
            None
        );
    }
}
