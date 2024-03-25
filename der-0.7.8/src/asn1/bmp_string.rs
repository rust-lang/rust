//! ASN.1 `BMPString` support.

use crate::{
    BytesOwned, DecodeValue, EncodeValue, Error, FixedTag, Header, Length, Reader, Result, Tag,
    Writer,
};
use alloc::{boxed::Box, vec::Vec};
use core::{fmt, str::FromStr};

/// ASN.1 `BMPString` type.
///
/// Encodes Basic Multilingual Plane (BMP) subset of Unicode (ISO 10646),
/// a.k.a. UCS-2.
#[derive(Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct BmpString {
    bytes: BytesOwned,
}

impl BmpString {
    /// Create a new [`BmpString`] from its UCS-2 encoding.
    pub fn from_ucs2(bytes: impl Into<Box<[u8]>>) -> Result<Self> {
        let bytes = bytes.into();

        if bytes.len() % 2 != 0 {
            return Err(Tag::BmpString.length_error());
        }

        let ret = Self {
            bytes: bytes.try_into()?,
        };

        for maybe_char in char::decode_utf16(ret.codepoints()) {
            match maybe_char {
                // All surrogates paired and character is in the Basic Multilingual Plane
                Ok(c) if (c as u64) < u64::from(u16::MAX) => (),
                // Unpaired surrogates or characters outside Basic Multilingual Plane
                _ => return Err(Tag::BmpString.value_error()),
            }
        }

        Ok(ret)
    }

    /// Create a new [`BmpString`] from a UTF-8 string.
    pub fn from_utf8(utf8: &str) -> Result<Self> {
        let capacity = utf8
            .len()
            .checked_mul(2)
            .ok_or_else(|| Tag::BmpString.length_error())?;

        let mut bytes = Vec::with_capacity(capacity);

        for code_point in utf8.encode_utf16() {
            bytes.extend(code_point.to_be_bytes());
        }

        Self::from_ucs2(bytes)
    }

    /// Borrow the encoded UCS-2 as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// Obtain the inner bytes.
    #[inline]
    pub fn into_bytes(self) -> Box<[u8]> {
        self.bytes.into()
    }

    /// Get an iterator over characters in the string.
    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        char::decode_utf16(self.codepoints())
            .map(|maybe_char| maybe_char.expect("unpaired surrogates checked in constructor"))
    }

    /// Get an iterator over the `u16` codepoints.
    pub fn codepoints(&self) -> impl Iterator<Item = u16> + '_ {
        // TODO(tarcieri): use `array_chunks`
        self.as_bytes()
            .chunks_exact(2)
            .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
    }
}

impl AsRef<[u8]> for BmpString {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> DecodeValue<'a> for BmpString {
    fn decode_value<R: Reader<'a>>(reader: &mut R, header: Header) -> Result<Self> {
        Self::from_ucs2(reader.read_vec(header.length)?)
    }
}

impl EncodeValue for BmpString {
    fn value_len(&self) -> Result<Length> {
        Ok(self.bytes.len())
    }

    fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
        writer.write(self.as_bytes())
    }
}

impl FixedTag for BmpString {
    const TAG: Tag = Tag::BmpString;
}

impl FromStr for BmpString {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::from_utf8(s)
    }
}

impl fmt::Debug for BmpString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BmpString(\"{}\")", self)
    }
}

impl fmt::Display for BmpString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in self.chars() {
            write!(f, "{}", c)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::BmpString;
    use crate::{Decode, Encode};
    use alloc::string::ToString;
    use hex_literal::hex;

    const EXAMPLE_BYTES: &[u8] = &hex!(
        "1e 26 00 43 00 65 00 72 00 74"
        "      00 69 00 66 00 69 00 63"
        "      00 61 00 74 00 65 00 54"
        "      00 65 00 6d 00 70 00 6c"
        "      00 61 00 74 00 65"
    );

    const EXAMPLE_UTF8: &str = "CertificateTemplate";

    #[test]
    fn decode() {
        let bmp_string = BmpString::from_der(EXAMPLE_BYTES).unwrap();
        assert_eq!(bmp_string.to_string(), EXAMPLE_UTF8);
    }

    #[test]
    fn encode() {
        let bmp_string = BmpString::from_utf8(EXAMPLE_UTF8).unwrap();
        let encoded = bmp_string.to_der().unwrap();
        assert_eq!(encoded, EXAMPLE_BYTES);
    }
}
