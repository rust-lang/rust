//! Slice reader.

use crate::{BytesRef, Decode, Error, ErrorKind, Header, Length, Reader, Result, Tag};

/// [`Reader`] which consumes an input byte slice.
#[derive(Clone, Debug)]
pub struct SliceReader<'a> {
    /// Byte slice being decoded.
    bytes: BytesRef<'a>,

    /// Did the decoding operation fail?
    failed: bool,

    /// Position within the decoded slice.
    position: Length,
}

impl<'a> SliceReader<'a> {
    /// Create a new slice reader for the given byte slice.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self {
            bytes: BytesRef::new(bytes)?,
            failed: false,
            position: Length::ZERO,
        })
    }

    /// Return an error with the given [`ErrorKind`], annotating it with
    /// context about where the error occurred.
    pub fn error(&mut self, kind: ErrorKind) -> Error {
        self.failed = true;
        kind.at(self.position)
    }

    /// Return an error for an invalid value with the given tag.
    pub fn value_error(&mut self, tag: Tag) -> Error {
        self.error(tag.value_error().kind())
    }

    /// Did the decoding operation fail due to an error?
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Obtain the remaining bytes in this slice reader from the current cursor
    /// position.
    fn remaining(&self) -> Result<&'a [u8]> {
        if self.is_failed() {
            Err(ErrorKind::Failed.at(self.position))
        } else {
            self.bytes
                .as_slice()
                .get(self.position.try_into()?..)
                .ok_or_else(|| Error::incomplete(self.input_len()))
        }
    }
}

impl<'a> Reader<'a> for SliceReader<'a> {
    fn input_len(&self) -> Length {
        self.bytes.len()
    }

    fn peek_byte(&self) -> Option<u8> {
        self.remaining()
            .ok()
            .and_then(|bytes| bytes.first().cloned())
    }

    fn peek_header(&self) -> Result<Header> {
        Header::decode(&mut self.clone())
    }

    fn position(&self) -> Length {
        self.position
    }

    fn read_slice(&mut self, len: Length) -> Result<&'a [u8]> {
        if self.is_failed() {
            return Err(self.error(ErrorKind::Failed));
        }

        match self.remaining()?.get(..len.try_into()?) {
            Some(result) => {
                self.position = (self.position + len)?;
                Ok(result)
            }
            None => Err(self.error(ErrorKind::Incomplete {
                expected_len: (self.position + len)?,
                actual_len: self.input_len(),
            })),
        }
    }

    fn decode<T: Decode<'a>>(&mut self) -> Result<T> {
        if self.is_failed() {
            return Err(self.error(ErrorKind::Failed));
        }

        T::decode(self).map_err(|e| {
            self.failed = true;
            e.nested(self.position)
        })
    }

    fn error(&mut self, kind: ErrorKind) -> Error {
        self.failed = true;
        kind.at(self.position)
    }

    fn finish<T>(self, value: T) -> Result<T> {
        if self.is_failed() {
            Err(ErrorKind::Failed.at(self.position))
        } else if !self.is_finished() {
            Err(ErrorKind::TrailingData {
                decoded: self.position,
                remaining: self.remaining_len(),
            }
            .at(self.position))
        } else {
            Ok(value)
        }
    }

    fn remaining_len(&self) -> Length {
        debug_assert!(self.position <= self.input_len());
        self.input_len().saturating_sub(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::SliceReader;
    use crate::{Decode, ErrorKind, Length, Reader, Tag};
    use hex_literal::hex;

    // INTEGER: 42
    const EXAMPLE_MSG: &[u8] = &hex!("02012A00");

    #[test]
    fn empty_message() {
        let mut reader = SliceReader::new(&[]).unwrap();
        let err = bool::decode(&mut reader).err().unwrap();
        assert_eq!(Some(Length::ZERO), err.position());

        match err.kind() {
            ErrorKind::Incomplete {
                expected_len,
                actual_len,
            } => {
                assert_eq!(actual_len, 0u8.into());
                assert_eq!(expected_len, 1u8.into());
            }
            other => panic!("unexpected error kind: {:?}", other),
        }
    }

    #[test]
    fn invalid_field_length() {
        const MSG_LEN: usize = 2;

        let mut reader = SliceReader::new(&EXAMPLE_MSG[..MSG_LEN]).unwrap();
        let err = i8::decode(&mut reader).err().unwrap();
        assert_eq!(Some(Length::from(2u8)), err.position());

        match err.kind() {
            ErrorKind::Incomplete {
                expected_len,
                actual_len,
            } => {
                assert_eq!(actual_len, MSG_LEN.try_into().unwrap());
                assert_eq!(expected_len, (MSG_LEN + 1).try_into().unwrap());
            }
            other => panic!("unexpected error kind: {:?}", other),
        }
    }

    #[test]
    fn trailing_data() {
        let mut reader = SliceReader::new(EXAMPLE_MSG).unwrap();
        let x = i8::decode(&mut reader).unwrap();
        assert_eq!(42i8, x);

        let err = reader.finish(x).err().unwrap();
        assert_eq!(Some(Length::from(3u8)), err.position());

        assert_eq!(
            ErrorKind::TrailingData {
                decoded: 3u8.into(),
                remaining: 1u8.into()
            },
            err.kind()
        );
    }

    #[test]
    fn peek_tag() {
        let reader = SliceReader::new(EXAMPLE_MSG).unwrap();
        assert_eq!(reader.position(), Length::ZERO);
        assert_eq!(reader.peek_tag().unwrap(), Tag::Integer);
        assert_eq!(reader.position(), Length::ZERO); // Position unchanged
    }

    #[test]
    fn peek_header() {
        let reader = SliceReader::new(EXAMPLE_MSG).unwrap();
        assert_eq!(reader.position(), Length::ZERO);

        let header = reader.peek_header().unwrap();
        assert_eq!(header.tag, Tag::Integer);
        assert_eq!(header.length, Length::ONE);
        assert_eq!(reader.position(), Length::ZERO); // Position unchanged
    }
}
