//! Streaming PEM reader.

use super::Reader;
use crate::{Decode, Error, ErrorKind, Header, Length, Result};
use core::cell::RefCell;

#[allow(clippy::integer_arithmetic)]
mod utils {
    use crate::{Error, Length, Result};
    use pem_rfc7468::Decoder;

    #[derive(Clone)]
    pub(super) struct BufReader<'i> {
        /// Inner PEM decoder.
        decoder: Decoder<'i>,

        /// Remaining after base64 decoding
        remaining: usize,

        /// Read buffer
        buf: [u8; BufReader::CAPACITY],

        /// Position of the head in the buffer,
        pos: usize,

        /// Position of the tail in the buffer,
        cap: usize,
    }

    impl<'i> BufReader<'i> {
        const CAPACITY: usize = 256;

        pub fn new(pem: &'i [u8]) -> Result<Self> {
            let decoder = Decoder::new(pem)?;
            let remaining = decoder.remaining_len();

            Ok(Self {
                decoder,
                remaining,
                buf: [0u8; 256],
                pos: 0,
                cap: 0,
            })
        }

        pub fn remaining_len(&self) -> usize {
            self.decoder.remaining_len() + self.cap - self.pos
        }

        fn fill_buffer(&mut self) -> Result<()> {
            debug_assert!(self.pos <= self.cap);

            if self.is_empty() {
                self.pos = 0;
                self.cap = 0;
            }

            let end = (self.cap + self.remaining).min(Self::CAPACITY);
            let writable_slice = &mut self.buf[self.cap..end];
            if writable_slice.is_empty() {
                return Ok(());
            }

            let wrote = self.decoder.decode(writable_slice)?.len();
            if wrote == 0 {
                return Err(Error::incomplete(Length::try_from(self.pos)?));
            }

            self.cap += wrote;
            self.remaining -= wrote;
            debug_assert!(self.cap <= Self::CAPACITY);

            Ok(())
        }

        /// Get the PEM label which will be used in the encapsulation boundaries
        /// for this document.
        pub fn type_label(&self) -> &'i str {
            self.decoder.type_label()
        }

        fn is_empty(&self) -> bool {
            self.pos == self.cap
        }

        fn as_slice(&self) -> &[u8] {
            &self.buf[self.pos..self.cap]
        }
    }

    impl<'i> BufReader<'i> {
        pub fn peek_byte(&self) -> Option<u8> {
            let s = self.as_slice();
            s.first().copied()
        }

        pub fn copy_to_slice<'o>(&mut self, buf: &'o mut [u8]) -> Result<&'o [u8]> {
            let mut output_pos = 0;

            while output_pos < buf.len() {
                if self.is_empty() {
                    self.fill_buffer()?;
                }

                let available = &self.buf[self.pos..self.cap];
                let window_len = (buf.len() - output_pos).min(available.len());
                let window = &mut buf[output_pos..output_pos + window_len];

                window.copy_from_slice(&available[..window_len]);
                self.pos += window_len;
                output_pos += window_len;
            }

            // Don't leave the read buffer empty for peek_byte()
            if self.is_empty() && self.decoder.remaining_len() != 0 {
                self.fill_buffer()?
            }

            debug_assert_eq!(output_pos, buf.len());

            Ok(buf)
        }
    }
}

/// `Reader` type which decodes PEM on-the-fly.
#[cfg(feature = "pem")]
#[derive(Clone)]
pub struct PemReader<'i> {
    /// Inner PEM decoder wrapped in a BufReader.
    reader: RefCell<utils::BufReader<'i>>,

    /// Input length (in bytes after Base64 decoding).
    input_len: Length,

    /// Position in the input buffer (in bytes after Base64 decoding).
    position: Length,
}

#[cfg(feature = "pem")]
impl<'i> PemReader<'i> {
    /// Create a new PEM reader which decodes data on-the-fly.
    ///
    /// Uses the default 64-character line wrapping.
    pub fn new(pem: &'i [u8]) -> Result<Self> {
        let reader = utils::BufReader::new(pem)?;
        let input_len = Length::try_from(reader.remaining_len())?;

        Ok(Self {
            reader: RefCell::new(reader),
            input_len,
            position: Length::ZERO,
        })
    }

    /// Get the PEM label which will be used in the encapsulation boundaries
    /// for this document.
    pub fn type_label(&self) -> &'i str {
        self.reader.borrow().type_label()
    }
}

#[cfg(feature = "pem")]
impl<'i> Reader<'i> for PemReader<'i> {
    fn input_len(&self) -> Length {
        self.input_len
    }

    fn peek_byte(&self) -> Option<u8> {
        if self.is_finished() {
            None
        } else {
            self.reader.borrow().peek_byte()
        }
    }

    fn peek_header(&self) -> Result<Header> {
        if self.is_finished() {
            Err(Error::incomplete(self.offset()))
        } else {
            Header::decode(&mut self.clone())
        }
    }

    fn position(&self) -> Length {
        self.position
    }

    fn read_slice(&mut self, _len: Length) -> Result<&'i [u8]> {
        // Can't borrow from PEM because it requires decoding
        Err(ErrorKind::Reader.into())
    }

    fn read_into<'o>(&mut self, buf: &'o mut [u8]) -> Result<&'o [u8]> {
        let bytes = self.reader.borrow_mut().copy_to_slice(buf)?;

        self.position = (self.position + bytes.len())?;

        debug_assert_eq!(
            self.position,
            (self.input_len - Length::try_from(self.reader.borrow().remaining_len())?)?
        );

        Ok(bytes)
    }
}
