//! Common handling for types backed by byte slices with enforcement of a
//! library-level length limitation i.e. `Length::max()`.

use crate::{
    DecodeValue, DerOrd, EncodeValue, Error, Header, Length, Reader, Result, StrRef, Writer,
};
use core::cmp::Ordering;

#[cfg(feature = "alloc")]
use crate::StrOwned;

/// Byte slice newtype which respects the `Length::max()` limit.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct BytesRef<'a> {
    /// Precomputed `Length` (avoids possible panicking conversions)
    pub length: Length,

    /// Inner value
    pub inner: &'a [u8],
}

impl<'a> BytesRef<'a> {
    /// Constant value representing an empty byte slice.
    pub const EMPTY: Self = Self {
        length: Length::ZERO,
        inner: &[],
    };

    /// Create a new [`BytesRef`], ensuring that the provided `slice` value
    /// is shorter than `Length::max()`.
    pub fn new(slice: &'a [u8]) -> Result<Self> {
        Ok(Self {
            length: Length::try_from(slice.len())?,
            inner: slice,
        })
    }

    /// Borrow the inner byte slice
    pub fn as_slice(&self) -> &'a [u8] {
        self.inner
    }

    /// Get the [`Length`] of this [`BytesRef`]
    pub fn len(self) -> Length {
        self.length
    }

    /// Is this [`BytesRef`] empty?
    pub fn is_empty(self) -> bool {
        self.len() == Length::ZERO
    }
}

impl AsRef<[u8]> for BytesRef<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<'a> DecodeValue<'a> for BytesRef<'a> {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        reader.read_slice(header.length).and_then(Self::new)
    }
}

impl EncodeValue for BytesRef<'_> {
    fn value_len(&self) -> Result<Length> {
        Ok(self.length)
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.as_ref())
    }
}

impl Default for BytesRef<'_> {
    fn default() -> Self {
        Self {
            length: Length::ZERO,
            inner: &[],
        }
    }
}

impl DerOrd for BytesRef<'_> {
    fn der_cmp(&self, other: &Self) -> Result<Ordering> {
        Ok(self.as_slice().cmp(other.as_slice()))
    }
}

impl<'a> From<StrRef<'a>> for BytesRef<'a> {
    fn from(s: StrRef<'a>) -> BytesRef<'a> {
        let bytes = s.as_bytes();
        debug_assert_eq!(bytes.len(), usize::try_from(s.length).expect("overflow"));

        BytesRef {
            inner: bytes,
            length: s.length,
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a> From<&'a StrOwned> for BytesRef<'a> {
    fn from(s: &'a StrOwned) -> BytesRef<'a> {
        let bytes = s.as_bytes();
        debug_assert_eq!(bytes.len(), usize::try_from(s.length).expect("overflow"));

        BytesRef {
            inner: bytes,
            length: s.length,
        }
    }
}

impl<'a> TryFrom<&'a [u8]> for BytesRef<'a> {
    type Error = Error;

    fn try_from(slice: &'a [u8]) -> Result<Self> {
        Self::new(slice)
    }
}

// Implement by hand because the derive would create invalid values.
// Make sure the length and the inner.len matches.
#[cfg(feature = "arbitrary")]
impl<'a> arbitrary::Arbitrary<'a> for BytesRef<'a> {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let length = u.arbitrary()?;
        Ok(Self {
            length,
            inner: u.bytes(u32::from(length) as usize)?,
        })
    }

    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        arbitrary::size_hint::and(Length::size_hint(depth), (0, None))
    }
}

#[cfg(feature = "alloc")]
mod allocating {
    use super::BytesRef;
    use crate::{referenced::RefToOwned, BytesOwned};

    impl<'a> RefToOwned<'a> for BytesRef<'a> {
        type Owned = BytesOwned;
        fn ref_to_owned(&self) -> Self::Owned {
            BytesOwned::from(*self)
        }
    }
}
