//! Reader type for consuming nested TLV records within a DER document.

use crate::{reader::Reader, Error, ErrorKind, Header, Length, Result};

/// Reader type used by [`Reader::read_nested`].
pub struct NestedReader<'i, R> {
    /// Inner reader type.
    inner: &'i mut R,

    /// Nested input length.
    input_len: Length,

    /// Position within the nested input.
    position: Length,
}

impl<'i, 'r, R: Reader<'r>> NestedReader<'i, R> {
    /// Create a new nested reader which can read the given [`Length`].
    pub(crate) fn new(inner: &'i mut R, len: Length) -> Result<Self> {
        if len <= inner.remaining_len() {
            Ok(Self {
                inner,
                input_len: len,
                position: Length::ZERO,
            })
        } else {
            Err(ErrorKind::Incomplete {
                expected_len: (inner.offset() + len)?,
                actual_len: (inner.offset() + inner.remaining_len())?,
            }
            .at(inner.offset()))
        }
    }

    /// Move the position cursor the given length, returning an error if there
    /// isn't enough remaining data in the nested input.
    fn advance_position(&mut self, len: Length) -> Result<()> {
        let new_position = (self.position + len)?;

        if new_position <= self.input_len {
            self.position = new_position;
            Ok(())
        } else {
            Err(ErrorKind::Incomplete {
                expected_len: (self.inner.offset() + len)?,
                actual_len: (self.inner.offset() + self.remaining_len())?,
            }
            .at(self.inner.offset()))
        }
    }
}

impl<'i, 'r, R: Reader<'r>> Reader<'r> for NestedReader<'i, R> {
    fn input_len(&self) -> Length {
        self.input_len
    }

    fn peek_byte(&self) -> Option<u8> {
        if self.is_finished() {
            None
        } else {
            self.inner.peek_byte()
        }
    }

    fn peek_header(&self) -> Result<Header> {
        if self.is_finished() {
            Err(Error::incomplete(self.offset()))
        } else {
            // TODO(tarcieri): handle peeking past nested length
            self.inner.peek_header()
        }
    }

    fn position(&self) -> Length {
        self.position
    }

    fn read_slice(&mut self, len: Length) -> Result<&'r [u8]> {
        self.advance_position(len)?;
        self.inner.read_slice(len)
    }

    fn error(&mut self, kind: ErrorKind) -> Error {
        self.inner.error(kind)
    }

    fn offset(&self) -> Length {
        self.inner.offset()
    }

    fn read_into<'o>(&mut self, out: &'o mut [u8]) -> Result<&'o [u8]> {
        self.advance_position(Length::try_from(out.len())?)?;
        self.inner.read_into(out)
    }
}
