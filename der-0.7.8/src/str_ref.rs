//! Common handling for types backed by `str` slices with enforcement of a
//! library-level length limitation i.e. `Length::max()`.

use crate::{BytesRef, DecodeValue, EncodeValue, Header, Length, Reader, Result, Writer};
use core::str;

/// String slice newtype which respects the [`Length::max`] limit.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct StrRef<'a> {
    /// Inner value
    pub(crate) inner: &'a str,

    /// Precomputed `Length` (avoids possible panicking conversions)
    pub(crate) length: Length,
}

impl<'a> StrRef<'a> {
    /// Create a new [`StrRef`], ensuring that the byte representation of
    /// the provided `str` value is shorter than `Length::max()`.
    pub fn new(s: &'a str) -> Result<Self> {
        Ok(Self {
            inner: s,
            length: Length::try_from(s.as_bytes().len())?,
        })
    }

    /// Parse a [`StrRef`] from UTF-8 encoded bytes.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        Self::new(str::from_utf8(bytes)?)
    }

    /// Borrow the inner `str`
    pub fn as_str(&self) -> &'a str {
        self.inner
    }

    /// Borrow the inner byte slice
    pub fn as_bytes(&self) -> &'a [u8] {
        self.inner.as_bytes()
    }

    /// Get the [`Length`] of this [`StrRef`]
    pub fn len(self) -> Length {
        self.length
    }

    /// Is this [`StrRef`] empty?
    pub fn is_empty(self) -> bool {
        self.len() == Length::ZERO
    }
}

impl AsRef<str> for StrRef<'_> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<[u8]> for StrRef<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> DecodeValue<'a> for StrRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        Self::from_bytes(BytesRef::decode_value(reader, header)?.as_slice())
    }
}

impl<'a> EncodeValue for StrRef<'a> {
    fn value_len(&self) -> Result<Length> {
        Ok(self.length)
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.as_ref())
    }
}

#[cfg(feature = "alloc")]
mod allocating {
    use super::StrRef;
    use crate::{referenced::RefToOwned, StrOwned};

    impl<'a> RefToOwned<'a> for StrRef<'a> {
        type Owned = StrOwned;
        fn ref_to_owned(&self) -> Self::Owned {
            StrOwned::from(*self)
        }
    }
}
