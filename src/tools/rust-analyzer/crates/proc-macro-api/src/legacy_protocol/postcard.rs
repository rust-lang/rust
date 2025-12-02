//! Postcard encode and decode implementations.

use std::io::{self, BufRead, Write};

use serde::{Serialize, de::DeserializeOwned};

use crate::{codec::Codec, framing::Framing};

pub struct PostcardProtocol;

impl Framing for PostcardProtocol {
    type Buf = Vec<u8>;

    fn read<'a, R: BufRead>(
        inp: &mut R,
        buf: &'a mut Vec<u8>,
    ) -> io::Result<Option<&'a mut Vec<u8>>> {
        buf.clear();
        let n = inp.read_until(0, buf)?;
        if n == 0 {
            return Ok(None);
        }
        Ok(Some(buf))
    }

    fn write<W: Write>(out: &mut W, buf: &Vec<u8>) -> io::Result<()> {
        out.write_all(buf)?;
        out.flush()
    }
}

impl Codec for PostcardProtocol {
    fn encode<T: Serialize>(msg: &T) -> io::Result<Vec<u8>> {
        postcard::to_allocvec_cobs(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    fn decode<T: DeserializeOwned>(buf: &mut Self::Buf) -> io::Result<T> {
        postcard::from_bytes_cobs(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
