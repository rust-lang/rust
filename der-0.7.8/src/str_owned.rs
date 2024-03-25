//! Common handling for types backed by `String` with enforcement of a
//! library-level length limitation i.e. `Length::max()`.

use crate::{
    referenced::OwnedToRef, BytesRef, DecodeValue, EncodeValue, Header, Length, Reader, Result,
    StrRef, Writer,
};
use alloc::string::String;
use core::str;

/// String newtype which respects the [`Length::max`] limit.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct StrOwned {
    /// Inner value
    pub(crate) inner: String,

    /// Precomputed `Length` (avoids possible panicking conversions)
    pub(crate) length: Length,
}

impl StrOwned {
    /// Create a new [`StrOwned`], ensuring that the byte representation of
    /// the provided `str` value is shorter than `Length::max()`.
    pub fn new(s: String) -> Result<Self> {
        let length = Length::try_from(s.as_bytes().len())?;

        Ok(Self { inner: s, length })
    }

    /// Parse a [`String`] from UTF-8 encoded bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            inner: String::from_utf8(bytes.to_vec())?,
            length: Length::try_from(bytes.len())?,
        })
    }

    /// Borrow the inner `str`
    pub fn as_str(&self) -> &str {
        &self.inner
    }

    /// Borrow the inner byte slice
    pub fn as_bytes(&self) -> &[u8] {
        self.inner.as_bytes()
    }

    /// Get the [`Length`] of this [`StrOwned`]
    pub fn len(&self) -> Length {
        self.length
    }

    /// Is this [`StrOwned`] empty?
    pub fn is_empty(&self) -> bool {
        self.len() == Length::ZERO
    }
}

impl AsRef<str> for StrOwned {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<[u8]> for StrOwned {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> DecodeValue<'a> for StrOwned {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        Self::from_bytes(BytesRef::decode_value(reader, header)?.as_slice())
    }
}

impl EncodeValue for StrOwned {
    fn value_len(&self) -> Result<Length> {
        Ok(self.length)
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.as_ref())
    }
}

impl From<StrRef<'_>> for StrOwned {
    fn from(s: StrRef<'_>) -> StrOwned {
        Self {
            inner: String::from(s.inner),
            length: s.length,
        }
    }
}

impl OwnedToRef for StrOwned {
    type Borrowed<'a> = StrRef<'a>;
    fn owned_to_ref(&self) -> Self::Borrowed<'_> {
        StrRef {
            length: self.length,
            inner: self.inner.as_ref(),
        }
    }
}
