//! Postcard encode and decode implementations.

use std::io::{self, BufRead, Write};

pub fn read_postcard<'a>(
    input: &mut impl BufRead,
    buf: &'a mut Vec<u8>,
) -> io::Result<Option<&'a mut [u8]>> {
    buf.clear();
    let n = input.read_until(0, buf)?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(&mut buf[..]))
}
pub fn write_postcard(out: &mut impl Write, msg: &[u8]) -> io::Result<()> {
    out.write_all(msg)?;
    out.flush()
}

pub fn encode_cobs<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, postcard::Error> {
    postcard::to_allocvec_cobs(value)
}

pub fn decode_cobs<T: serde::de::DeserializeOwned>(bytes: &mut [u8]) -> Result<T, postcard::Error> {
    postcard::from_bytes_cobs(bytes)
}
