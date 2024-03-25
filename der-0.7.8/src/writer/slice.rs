//! Slice writer.

use crate::{
    asn1::*, Encode, EncodeValue, ErrorKind, Header, Length, Result, Tag, TagMode, TagNumber,
    Tagged, Writer,
};

/// [`Writer`] which encodes DER into a mutable output byte slice.
#[derive(Debug)]
pub struct SliceWriter<'a> {
    /// Buffer into which DER-encoded message is written
    bytes: &'a mut [u8],

    /// Has the encoding operation failed?
    failed: bool,

    /// Total number of bytes written to buffer so far
    position: Length,
}

impl<'a> SliceWriter<'a> {
    /// Create a new encoder with the given byte slice as a backing buffer.
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            failed: false,
            position: Length::ZERO,
        }
    }

    /// Encode a value which impls the [`Encode`] trait.
    pub fn encode<T: Encode>(&mut self, encodable: &T) -> Result<()> {
        if self.is_failed() {
            self.error(ErrorKind::Failed)?
        }

        encodable.encode(self).map_err(|e| {
            self.failed = true;
            e.nested(self.position)
        })
    }

    /// Return an error with the given [`ErrorKind`], annotating it with
    /// context about where the error occurred.
    pub fn error<T>(&mut self, kind: ErrorKind) -> Result<T> {
        self.failed = true;
        Err(kind.at(self.position))
    }

    /// Did the decoding operation fail due to an error?
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Finish encoding to the buffer, returning a slice containing the data
    /// written to the buffer.
    pub fn finish(self) -> Result<&'a [u8]> {
        let position = self.position;

        if self.is_failed() {
            return Err(ErrorKind::Failed.at(position));
        }

        self.bytes
            .get(..usize::try_from(position)?)
            .ok_or_else(|| ErrorKind::Overlength.at(position))
    }

    /// Encode a `CONTEXT-SPECIFIC` field with the provided tag number and mode.
    pub fn context_specific<T>(
        &mut self,
        tag_number: TagNumber,
        tag_mode: TagMode,
        value: &T,
    ) -> Result<()>
    where
        T: EncodeValue + Tagged,
    {
        ContextSpecificRef {
            tag_number,
            tag_mode,
            value,
        }
        .encode(self)
    }

    /// Encode an ASN.1 `SEQUENCE` of the given length.
    ///
    /// Spawns a nested slice writer which is expected to be exactly the
    /// specified length upon completion.
    pub fn sequence<F>(&mut self, length: Length, f: F) -> Result<()>
    where
        F: FnOnce(&mut SliceWriter<'_>) -> Result<()>,
    {
        Header::new(Tag::Sequence, length).and_then(|header| header.encode(self))?;

        let mut nested_encoder = SliceWriter::new(self.reserve(length)?);
        f(&mut nested_encoder)?;

        if nested_encoder.finish()?.len() == usize::try_from(length)? {
            Ok(())
        } else {
            self.error(ErrorKind::Length { tag: Tag::Sequence })
        }
    }

    /// Reserve a portion of the internal buffer, updating the internal cursor
    /// position and returning a mutable slice.
    fn reserve(&mut self, len: impl TryInto<Length>) -> Result<&mut [u8]> {
        if self.is_failed() {
            return Err(ErrorKind::Failed.at(self.position));
        }

        let len = len
            .try_into()
            .or_else(|_| self.error(ErrorKind::Overflow))?;

        let end = (self.position + len).or_else(|e| self.error(e.kind()))?;
        let slice = self
            .bytes
            .get_mut(self.position.try_into()?..end.try_into()?)
            .ok_or_else(|| ErrorKind::Overlength.at(end))?;

        self.position = end;
        Ok(slice)
    }
}

impl<'a> Writer for SliceWriter<'a> {
    fn write(&mut self, slice: &[u8]) -> Result<()> {
        self.reserve(slice.len())?.copy_from_slice(slice);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SliceWriter;
    use crate::{Encode, ErrorKind, Length};

    #[test]
    fn overlength_message() {
        let mut buffer = [];
        let mut writer = SliceWriter::new(&mut buffer);
        let err = false.encode(&mut writer).err().unwrap();
        assert_eq!(err.kind(), ErrorKind::Overlength);
        assert_eq!(err.position(), Some(Length::ONE));
    }
}
